"""
Douban (movie) vLLM batch inference client -- Llama-3-8B variant.

Run, e.g.:
    python3 s21_inference_vllm_llama3_douban.py                       
    SEED=1 OUTPUT_DIR=llama_3_8b_output/douban_movies/run_1 python3 s21_inference_vllm_llama3_douban.py
    SEED=2 OUTPUT_DIR=llama_3_8b_output/douban_movies/run_2 python3 s21_inference_vllm_llama3_douban.py
    SEED=3 OUTPUT_DIR=llama_3_8b_output/douban_movies/run_3 python3 s21_inference_vllm_llama3_douban.py
"""
import csv
import json
import os
import re
import numpy as np
import pandas as pd
import requests as http
from transformers import AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────
RETRIEVAL_CSV_TEMPLATE = (
    "0911_douban_test_retrieved_format/retrieval_context_per_query_embed_rerank_k{k}_a50_b50_sig1_gtcase1_gtprior0_simscore0.csv"
)
_TREND_ARRAY_SPLIT_PATTERN = re.compile(r"[,\s]+")

K_VALUES = [1,5,10]

# ── Model / backend config ──────────────────────────────────────────────
MODEL = "llama-3-8b-instruct"    # must match the --model/positional arg vLLM was started with
# VLLM_HOST = "http://localhost:8001"
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_HOST = f"http://localhost:{VLLM_PORT}"

# Per-request RNG seed forwarded to vLLM -- set via `SEED=<int> python3 ...`.
# Left unset (None), vLLM uses its default RNG state, which is typically the
# SAME across independent server launches: replicate runs will come back
# byte-identical despite temperature=0.1. Pass a DIFFERENT SEED per
# replicate run (1, 2, 3, ...) to get genuine run-to-run variance.
SEED = os.environ.get("SEED")
SEED = int(SEED) if SEED not in (None, "") else None
if SEED is None:
    print("WARNING: SEED env var not set -- vLLM will use its default RNG state, "
          "which is typically the SAME across independent server launches. If "
          "this is one of several replicate runs, set SEED=1/2/3/... per run "
          "(and OUTPUT_DIR to match) or they will come back byte-identical.")

# OUTPUT_JSON_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "reasoning": {
#             "type": "string",
#             "minLength": 150,
#             "maxLength": 1400,  # character limit, not token count -- see note below
#         },
#         "predict_popularity_score": {
#             "type": "string",
#             "pattern": "^[0-9]+$",
#         },
#     },
#     "required": ["reasoning", "predict_popularity_score"],
#     "additionalProperties": False,
# }

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "minLength": 150,
            "maxLength": 1400,
        },
        "predict_popularity_score": {
            "type": "string",
            "pattern": "^[0-9]+(\\.[0-9]+)?$",
        },
    },
    "required": ["reasoning", "predict_popularity_score"],
    "additionalProperties": False,
}

# Overridable via `OUTPUT_DIR=<path> python3 ...` (same pattern as SEED/
# VLLM_PORT above) so a PBS job script can point each replicate run at its
# own folder via `qsub -v OUTPUT_DIR=...` without hand-editing this file
# between submissions.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "llama_3_8b_output/douban_movies/run_1")
NO_RETRIEVAL_FALLBACK = "No similar past items provided."
INFERENCE_MONTH_STR = "2019-11"  # test-set prediction time, fixed for this whole run
# SMOKE_TEST_MOVIE_IDS = [
#     27119724, 30166972, 30402296, 27109633, 25887288,
#     30299311, 26786669, 30318116, 30362186, 33400376,
#     30390144, 3097572, 26794435, 30413052, 26612258,
#     27010768, 6981153, 26715636, 30272143, 25779217, 30295905, 32659890,
#     26235346, 26426056, 27595272, 30327842, 30282387
# ]


_tokenizer = AutoTokenizer.from_pretrained(MODEL)

def build_emoji_logit_bias(tokenizer, bias_value=-100):
    """Ban the byte-fallback tokens that start a 4-byte UTF-8 sequence
    (lead bytes 0xF0-0xF4). NOTE: this was written for Llama-2's
    SentencePiece tokenizer, which represents unseen Unicode as explicit
    '<0xNN>' byte tokens. Llama-3 uses a different (tiktoken-style BPE,
    128k-vocab) tokenizer that may not expose the same '<0xF0>'-style token
    strings -- if none of these are found in Llama-3's vocab, `bias` will
    come back empty and this mechanism will do nothing (silently, not an
    error). See the startup warning below."""
    vocab = tokenizer.get_vocab()
    bias = {}
    for lead_byte in ["0xF0", "0xF1", "0xF2", "0xF3", "0xF4"]:
        token_str = f"<{lead_byte}>"
        if token_str in vocab:
            bias[str(vocab[token_str])] = bias_value
    return bias

EMOJI_LOGIT_BIAS = build_emoji_logit_bias(_tokenizer)
if not EMOJI_LOGIT_BIAS:
    print("WARNING: EMOJI_LOGIT_BIAS is empty -- Llama-3's tokenizer doesn't "
          "expose '<0xF0>'-style byte tokens the way Llama-2's does. Emoji "
          "banning via logit_bias will have NO effect for this model. If "
          "emoji/symbol issues show up in Llama-3 outputs, this needs a "
          "different approach (e.g. banning specific emoji token ids "
          "directly, found by tokenizing sample emoji and inspecting ids).")
# print(EMOJI_LOGIT_BIAS)
# ── END CONFIG ────────────────────────────────────────────────────────────


def parse_center_name(center_name):
    """'high_popular__decrease' -> ('high_popular', 'decrease'). Returns
    (None, None) for anything unparseable (e.g. a scoring error row)."""
    if not isinstance(center_name, str) or "__" not in center_name:
        return None, None
    pop_cluster, direction = center_name.split("__", 1)
    return pop_cluster, direction


def sample_equal_per_direction(
    df: pd.DataFrame,
    target_n: int,
    target_high_pop: int = 10,
    cluster_col: str = 'ground_truth_pop_cluster',
    high_pop_val: str = 'high_popular',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Samples target_n rows equalized across ground_truth_direction,
    enforcing exactly `target_high_pop` total rows with high popularity cluster.
    """
    directions = ['increase', 'decrease', 'stable']
    n_dirs = len(directions)

    high_per_dir = target_high_pop // n_dirs
    high_remainder = target_high_pop % n_dirs

    tot_per_dir = target_n // n_dirs
    tot_remainder = target_n % n_dirs

    sampled_chunks = []

    for i, d in enumerate(directions):
        n_high_needed = high_per_dir + (1 if i < high_remainder else 0)
        n_total_needed = tot_per_dir + (1 if i < tot_remainder else 0)
        n_normal_needed = n_total_needed - n_high_needed

        sub_dir = df[df['ground_truth_direction'] == d]

        sub_high = sub_dir[sub_dir[cluster_col] == high_pop_val]
        sub_normal = sub_dir[sub_dir[cluster_col] != high_pop_val]

        if len(sub_high) <= n_high_needed:
            sampled_high = sub_high
        else:
            sampled_high = sub_high.sample(n=n_high_needed, random_state=random_state)

        if len(sub_normal) <= n_normal_needed:
            sampled_normal = sub_normal
        else:
            sampled_normal = sub_normal.sample(n=n_normal_needed, random_state=random_state)

        sampled_chunks.extend([sampled_high, sampled_normal])

    return pd.concat(sampled_chunks).reset_index(drop=True)


def load_retrieval_csv(k):
    path = RETRIEVAL_CSV_TEMPLATE.format(k=k)
    df = pd.read_csv(path)

    df["trend_array"] = df["trend_array"].apply(
        lambda x: np.array([int(i) for i in _TREND_ARRAY_SPLIT_PATTERN.split(x.strip("[]").strip()) if i])
        if isinstance(x, str) else x
    )

    return df


# ── Movie description helpers ─────────────────────────────────────

def _fallback_na(value):
    """Return 'NA' for missing/empty values, else the value as-is."""
    if value is None:
        return "NA"
    if isinstance(value, float) and pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return "NA"
    return text


def capitalize_names(text):
    """Title-case a comma-separated list of person names, e.g.
    'frank miller,robert rodriguez' -> 'Frank Miller, Robert Rodriguez'."""
    if not isinstance(text, str) or text.strip().upper() == "NA":
        return text
    names = [name.strip().title() for name in text.split(",")]
    return ", ".join(name for name in names if name)


def build_movie_description(row):
    """Build the MovieDescription text from the structured columns."""
    movie_name = _fallback_na(row.get("movie_name"))
    release_date = _fallback_na(row.get("release_date"))
    genre = _fallback_na(row.get("genre"))
    synopsis = _fallback_na(row.get("description"))
    language = _fallback_na(row.get("language"))
    director = capitalize_names(_fallback_na(row.get("director")))
    writers = capitalize_names(_fallback_na(row.get("writers")))
    top5_actors = _fallback_na(row.get("top5_actors"))
    producers = _fallback_na(row.get("producers"))
    music = _fallback_na(row.get("music"))  # no music/composer column available in this pipeline

    return (
        f"The movie {movie_name} is a {genre} film released in {release_date}. "
        f"In this movie, {synopsis}. "
        f"It is in {language}, directed by {director}, and written by {writers}. "
        f"And the main actors (top-5) include {top5_actors}. "
        f"The movie is produced by {producers} and the music is produced by {music}."
    )

def _month_label(date_value):
    """'YYYY-MM' label for a date-like value, or None if missing/unparsable."""
    if pd.isna(date_value):
        return None
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M"))
    except (ValueError, TypeError):
        return None

def _month_before(date_value):
    """'YYYY-MM' label for the calendar month immediately preceding date_value."""
    if pd.isna(date_value):
        return None
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M") - 1)
    except (ValueError, TypeError):
        return None

def build_target_trend_line(trend_array, release_date, inference_month_str):
    """Same rendering as the retrieved-item cards' Trend line -- a date
    window (release month through the month before InferenceTimestamp)
    plus a plain comma-separated value sequence -- built from trend_array
    instead of the colon-string. Matching the retrieved items' format here
    matters: the old {timestamp, popularity_score} dict rendering was
    implicated in own_estimate just echoing the last listed value rather
    than reasoning over the trend, and giving the target a differently-shaped
    history than its comps risked the model treating them as two different
    kinds of object instead of the same kind of sequence.

    Returns None for a debut item (empty or [-1]-sentinel trend_array) --
    callers should render an explicit "No historical trend" note in that
    case rather than an empty list literal.
    """
    trend_list = trend_array.tolist() if hasattr(trend_array, "tolist") else list(trend_array)
    if len(trend_list) == 0 or (len(trend_list) == 1 and trend_list[0] == -1):
        return None

    start_label = _month_label(release_date)
    end_label = _month_before(inference_month_str)

    if start_label and end_label:
        window = f", {start_label}" if start_label == end_label else f", {start_label} to {end_label}"
        window += " (monthly)"
    else:
        window = ""
    values_str = ", ".join(map(str, trend_list))
    label = f"Trend{window}" if window else "Trend (full history)"

    return f"{label}: {values_str}"

def build_target_item_block(row, inference_month_str):
    movie_description_text = build_movie_description(row)

    if row['item_age'] == -1:
        # Debut item -- no release history yet, same convention as the
        # retrieved-item debut cards (which omit the Trend line entirely).
        popularity_history_text = "Debut item — no historical trend available"
    else:
        trend_line = build_target_trend_line(
            row['trend_array'], row.get('release_date'), inference_month_str
        )
        popularity_history_text = trend_line if trend_line is not None else "No historical trend"

    return f"""
<MovieDescription>{movie_description_text}</MovieDescription>

<PopularityHistory>{popularity_history_text}</PopularityHistory>

<InferenceTimestamp>{inference_month_str}</InferenceTimestamp>
"""


SYSTEM_PROMPT = """You are an expert movie popularity analyst. Your task is to predict a movie's popularity score at a given point in time, based on its own
descriptive features, its historical popularity trajectory (when available), and a set of retrieved past movies with similar description and popularity patterns.

### About <PopularityHistory>:
<PopularityHistory> covers the target's own monthly popularity values from
release through one month before <InferenceTimestamp> -- it does not
include the value at <InferenceTimestamp> itself, which is what you are
being asked to predict.

### About <SimilarPastItems>:
<SimilarPastItems> shows past movies retrieved for similar description and
popularity pattern -- they will not necessarily match the target's outcome.
For non-debut items, each item's own trend line covers its own history up through one month before its own outcome month; its outcome is stated only as a relative percentage change from that trend's last value (never the raw outcome number itself), showing how much the observed popularity moved.
For debut items, neither a trend line nor an outcome value is provided, since there is no prior popularity history — only its own movie description and explanatory factors are available for these cases.

Steps:
1. First, estimate the target's popularity score using only <MovieDescription>
   and <PopularityHistory> (when available). It's a specific number, not a category label.
2. Then review <SimilarPastItems>. For each retrieved item, check whether
   its trend trajectory and explanatory factors -- covering both its own
   movie description and its own popularity history -- are actually
   applicable to the target item. Use this review to adjust your step 1
   estimate up, down, or keep it stable. Repeated factors across items
   count as one signal, not reinforcement.

Output a single, valid JSON object with exactly these two keys:

- `reasoning`: must cover two things, in order -- (1) your step 1 initial
  estimate and what it's based on, and (2) your justification for the
  final predict_popularity_score, including whether and why
  <SimilarPastItems> moved it up, down, or left it unchanged from that
  initial estimate.
- `predict_popularity_score`: your final predicted popularity score at the
  given <InferenceTimestamp>. A non-negative number, formatted as a string
  (decimals are allowed, do not strip the decimal point or round to an integer).

Output only this JSON object, with nothing before or after.

### Input format:

<MovieDescription>...</MovieDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
<SimilarPastItems>...</SimilarPastItems>
"""
def prepare_batch_input(df, model, variant_tag, inference_month_str,
                         include_retrieval=True, output_dir=".", seed=SEED):
    print(f"Preparing batch: variant={variant_tag} (include_retrieval={include_retrieval}), {len(df)} rows")

    batch_requests = []
    custom_ids = []

    for idx, row in df.iterrows():
        custom_id = f"pred_{variant_tag}_request-{idx + 1:05d}"
        custom_ids.append(custom_id)

        if include_retrieval:
            retrieved_section = f"<SimilarPastItems>\n{row['retrieved_context']}\n</SimilarPastItems>"
        else:
            retrieved_section = f"<SimilarPastItems>\n{NO_RETRIEVAL_FALLBACK}\n</SimilarPastItems>"

        target_item_block = build_target_item_block(row, inference_month_str)

        user_content = f"""{target_item_block}

        {retrieved_section}
        """
        # Llama-3-8B's own native (non-extended) context window is 8192 --
        # matches Llama-2's budget without needing to cap anything down.
        MAX_MODEL_LEN = 8192
        SAFETY_MARGIN = 50
        MAX_TOKENS_CEILING = 1600  # raised from 1000 -- see note below

        prompt_tokens = len(_tokenizer.encode(SYSTEM_PROMPT + user_content))
        max_tokens = min(MAX_TOKENS_CEILING, MAX_MODEL_LEN - prompt_tokens - SAFETY_MARGIN)
        if max_tokens < 150:
            print(f"WARNING: {custom_id} has very little completion room ({max_tokens} tokens, prompt={prompt_tokens}) — may truncate or fail")
            max_tokens = max(max_tokens, 1)

        print(f"{custom_id}: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "top_p": 1,
            "logit_bias": EMOJI_LOGIT_BIAS,
            "repetition_penalty": 1.15,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "movie_popularity_prediction",
                    "schema": OUTPUT_JSON_SCHEMA,
                },
            },
        }

        # Per-request RNG seed -- see the SEED config note above. Passed
        # through unchanged from whatever `seed` this function was called
        # with (defaults to the module-level SEED).
        if seed is not None:
            body["seed"] = seed

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        batch_requests.append(request)

    mapping_df = df.copy()
    mapping_df["custom_id"] = custom_ids
    mapping_df["seed"] = seed  # audit trail: which seed produced this run's predictions
    if not include_retrieval and "retrieved_context" in mapping_df.columns:
        mapping_df = mapping_df.drop(columns=["retrieved_context"])

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = f"{output_dir}/mapping_{variant_tag}.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping ({len(mapping_df)} rows) -> {mapping_path}")
    print(f"Prepared {len(batch_requests)} requests")
    return batch_requests

def run_via_vllm(batch_requests, variant_tag, output_dir=".", host=VLLM_HOST):
    """Sends each request to a running vLLM server and saves results to CSV
    incrementally (one row at a time), so progress survives a crash/interrupt.
    Skips custom_ids already present in an existing output file, so a
    re-run after a failure resumes instead of starting over."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/test_pred_results_{variant_tag}.csv"

    # Resume support: figure out what's already done
    done_ids = set()
    file_exists = os.path.isfile(out_path)
    if file_exists:
        existing = pd.read_csv(out_path)
        done_ids = set(existing["custom_id"])
        print(f"Resuming: {len(done_ids)} rows already completed, skipping those.")

    fieldnames = ["custom_id", "status_code", "content", "error"]
    n_ok, n_err, n_skipped = 0, 0, 0

    # Open in append mode; write header only if file is new
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, req in enumerate(batch_requests, 1):
            if req["custom_id"] in done_ids:
                n_skipped += 1
                continue

            try:
                resp = http.post(f"{host}/v1/chat/completions", json=req["body"], timeout=300)
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                row = {"custom_id": req["custom_id"], "status_code": resp.status_code,
                       "content": content, "error": None}
                n_ok += 1
            except http.exceptions.HTTPError:
                row = {"custom_id": req["custom_id"], "status_code": resp.status_code,
                       "content": None, "error": resp.text}
                n_err += 1
            except Exception as e:
                row = {"custom_id": req["custom_id"], "status_code": None,
                       "content": None, "error": str(e)}
                n_err += 1

            writer.writerow(row)
            f.flush()  # force it to disk immediately, don't wait for buffer

            if i % 10 == 0 or i == len(batch_requests):
                print(f"  [{variant_tag}] {i}/{len(batch_requests)} done (ok={n_ok}, err={n_err}, skipped={n_skipped})")

    print(f"Saved results -> {out_path} (ok={n_ok}, err={n_err}, skipped={n_skipped})")
    return out_path


# def is_valid_result(content: str, inference_month_str: str = INFERENCE_MONTH_STR) -> bool:
#     if not content or not isinstance(content, str):
#         return False
#     try:
#         obj = json.loads(content)
#     except (json.JSONDecodeError, ValueError):
#         return False
#     if not isinstance(obj, dict):
#         return False
#     reasoning = obj.get("reasoning")
#     score = obj.get("predict_popularity_score")
#     if not reasoning or not isinstance(reasoning, str) or len(reasoning.strip()) < 30:
#         return False
#     if score is None:
#         return False
#     score_str = str(score).strip()
#     if not score_str.isdigit():
#         return False
#     stripped_timestamp = inference_month_str.replace("-", "")
#     if score_str == stripped_timestamp:
#         return False
#     if len(score_str) > 5:
#         return False
#     return True

def is_valid_result(content: str, inference_month_str: str = INFERENCE_MONTH_STR) -> bool:
    if not content or not isinstance(content, str):
        return False
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(obj, dict):
        return False
    reasoning = obj.get("reasoning")
    score = obj.get("predict_popularity_score")
    if not reasoning or not isinstance(reasoning, str) or len(reasoning.strip()) < 30:
        return False
    if score is None:
        return False
    score_str = str(score).strip()
    try:
        score_val = float(score_str)
    except ValueError:
        return False
    stripped_timestamp = inference_month_str.replace("-", "")
    if str(int(round(score_val))) == stripped_timestamp:
        return False
    if len(str(int(score_val))) > 5:
        return False
    return True


def repair_incomplete_rows(
    out_path: str,
    batch_requests_by_id: dict,
    host: str = VLLM_HOST,
    max_retries: int = 3,
):
    """Reread out_path, find rows that failed to produce valid JSON,
    resend just those requests, and rewrite the CSV in place.

    Every row gets a `was_repaired` (bool) and `repair_temperature` column
    written -- whether the repair attempt succeeded or not -- so there's a
    record of which rows needed a retry and at what temperature the fix
    landed, instead of silently overwriting content with no trace a repair
    happened."""
    df = pd.read_csv(out_path)

    if "was_repaired" not in df.columns:
        df["was_repaired"] = False
    if "repair_temperature" not in df.columns:
        df["repair_temperature"] = None

    bad_mask = ~df["content"].apply(is_valid_result)
    bad_ids = df.loc[bad_mask, "custom_id"].tolist()
    print(f"{len(bad_ids)} / {len(df)} rows need repair")

    for custom_id in bad_ids:
        if custom_id not in batch_requests_by_id:
            print(f"  skip {custom_id}: no original request found")
            continue

        req = batch_requests_by_id[custom_id]
        fixed_content = None
        fixed_status = None
        fixed_temperature = None

        for attempt in range(1, max_retries + 1):
            retry_body = json.loads(json.dumps(req["body"]))  # deep copy
            retry_body["temperature"] = min(0.2 + 0.2 * attempt, 0.8)
            retry_body["repetition_penalty"] = 1.3

            try:
                resp = http.post(f"{host}/v1/chat/completions", json=retry_body, timeout=300)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"  {custom_id} attempt {attempt} errored: {e}")
                continue

            if is_valid_result(content):
                fixed_content = content
                fixed_status = resp.status_code
                fixed_temperature = retry_body["temperature"]
                print(f"  {custom_id} fixed on attempt {attempt}")
                break
            else:
                print(f"  {custom_id} attempt {attempt} still invalid")

        idx = df.index[df["custom_id"] == custom_id][0]
        if fixed_content is not None:
            df.at[idx, "content"] = fixed_content
            df.at[idx, "status_code"] = fixed_status
            df.at[idx, "error"] = None
            df.at[idx, "was_repaired"] = True
            df.at[idx, "repair_temperature"] = fixed_temperature
        else:
            print(f"  {custom_id} still bad after {max_retries} attempts — leaving as-is for manual review")
            df.at[idx, "was_repaired"] = True  # attempted but not fixed

    df.to_csv(out_path, index=False)
    still_bad = (~df["content"].apply(is_valid_result)).sum()
    print(f"Repair pass done. {still_bad} rows remain invalid.")

def main():
    for k in K_VALUES:
        df = load_retrieval_csv(k)
        # df = sample_equal_per_direction(df, 80, target_high_pop=10).copy()
        variant_tag = f"k{k}_full"

        requests_k = prepare_batch_input(
            df, MODEL, variant_tag, INFERENCE_MONTH_STR,
            include_retrieval=True, output_dir=OUTPUT_DIR,
        )
        run_via_vllm(requests_k, variant_tag, OUTPUT_DIR)

        # NEW: repair pass on this variant's output
        batch_requests_by_id = {req["custom_id"]: req for req in requests_k}
        repair_incomplete_rows(
            out_path=f"{OUTPUT_DIR}/test_pred_results_{variant_tag}.csv",
            batch_requests_by_id=batch_requests_by_id,
        )

    df_no_retrieval = load_retrieval_csv(K_VALUES[0])
    # subset_no_retrieval = sample_equal_per_direction(df_no_retrieval, 80, target_high_pop=10)
    df_no_retrieval = df_no_retrieval.drop(columns=['retrieved_context']).copy()
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, MODEL, "no_retrieval", INFERENCE_MONTH_STR,
        include_retrieval=False, output_dir=OUTPUT_DIR,
    )
    run_via_vllm(requests_no_retrieval, "no_retrieval", OUTPUT_DIR)

    # NEW: repair pass on the no-retrieval variant's output
    batch_requests_by_id_no_retrieval = {req["custom_id"]: req for req in requests_no_retrieval}
    repair_incomplete_rows(
        out_path=f"{OUTPUT_DIR}/test_pred_results_no_retrieval.csv",
        batch_requests_by_id=batch_requests_by_id_no_retrieval,
    )

if __name__ == "__main__":
    main()
