"""
Douban (movie) inference client -- DeepSeek variant.

Run, e.g.:
    export DEEPSEEK_API_KEY=sk-...
    OUTPUT_DIR=deepseek_flash_output/douban_movies/run_1 python3 s21_inference_deepseek_douban.py
    OUTPUT_DIR=deepseek_flash_output/douban_movies/run_2 python3 s21_inference_deepseek_douban.py
    OUTPUT_DIR=deepseek_flash_output/douban_movies/run_3 python3 s21_inference_deepseek_douban.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

from deepseek_client import (
    DEEPSEEK_MODEL,
    MAX_TOKENS_CEILING,
    build_deepseek_body,
    get_api_key,
    run_via_deepseek,
    repair_incomplete_rows_deepseek,
)

# ── CONFIG ────────────────────────────────────────────────────────────────
RETRIEVAL_CSV_TEMPLATE = (
    "0911_douban_test_retrieved_format/retrieval_context_per_query_embed_rerank_k{k}_a50_b50_sig1_gtcase1_gtprior0_simscore0.csv"
)
_TREND_ARRAY_SPLIT_PATTERN = re.compile(r"[,\s]+")

K_VALUES = [1, 5, 10]

# Overridable via `OUTPUT_DIR=<path> python3 ...`, same pattern as the
# vLLM Douban scripts, so each replicate run gets its own folder. No SEED
# env var here -- not supported by DeepSeek's API, and not needed (see
# module docstring).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "deepseek_flash_output/douban_movies/run_1")
NO_RETRIEVAL_FALLBACK = "No similar past items provided."
INFERENCE_MONTH_STR = "2019-11"  # test-set prediction time, fixed for this whole run
# ── END CONFIG ────────────────────────────────────────────────────────────


def load_retrieval_csv(k):
    path = RETRIEVAL_CSV_TEMPLATE.format(k=k)
    df = pd.read_csv(path)

    df["trend_array"] = df["trend_array"].apply(
        lambda x: np.array([int(i) for i in _TREND_ARRAY_SPLIT_PATTERN.split(x.strip("[]").strip()) if i])
        if isinstance(x, str) else x
    )

    return df


# ── Friend-format alignment helpers (unchanged from s21_inference_vllm_qwen_douban.py) ──

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
    """Reproduce the friend's exact MovieDescription template from our
    structured columns."""
    movie_name = _fallback_na(row.get("movie_name"))
    release_date = _fallback_na(row.get("release_date"))
    genre = _fallback_na(row.get("genre"))
    synopsis = _fallback_na(row.get("description"))
    language = _fallback_na(row.get("language"))
    director = capitalize_names(_fallback_na(row.get("director")))
    writers = capitalize_names(_fallback_na(row.get("writers")))
    top5_actors = _fallback_na(row.get("top5_actors"))
    producers = _fallback_na(row.get("producers"))
    music = _fallback_na(row.get("music"))

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
    """Same rendering as the retrieved-item cards' Trend line. Returns None
    for a debut item (empty or [-1]-sentinel trend_array)."""
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
  given <InferenceTimestamp>. Non-negative integer, formatted as a string.

Output only this JSON object, with nothing before or after.

### Input format:

<MovieDescription>...</MovieDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
<SimilarPastItems>...</SimilarPastItems>
"""


def prepare_batch_input(df, variant_tag, inference_month_str, include_retrieval=True, output_dir="."):
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

        body = build_deepseek_body(SYSTEM_PROMPT, user_content, max_tokens=MAX_TOKENS_CEILING)

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": body,
        }
        batch_requests.append(request)

    mapping_df = df.copy()
    mapping_df["custom_id"] = custom_ids
    mapping_df["model"] = DEEPSEEK_MODEL  # audit trail (no seed column -- not supported/applicable here)
    if not include_retrieval and "retrieved_context" in mapping_df.columns:
        mapping_df = mapping_df.drop(columns=["retrieved_context"])

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = f"{output_dir}/mapping_{variant_tag}.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping ({len(mapping_df)} rows) -> {mapping_path}")
    print(f"Prepared {len(batch_requests)} requests")
    return batch_requests


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
    if not score_str.isdigit():
        return False
    stripped_timestamp = inference_month_str.replace("-", "")
    if score_str == stripped_timestamp:
        return False
    if len(score_str) > 5:
        return False
    return True


def main():
    api_key = get_api_key()
    print(f"model={DEEPSEEK_MODEL}  output_dir={OUTPUT_DIR}")

    for k in K_VALUES:
        df = load_retrieval_csv(k)
        variant_tag = f"k{k}_full"

        requests_k = prepare_batch_input(
            df, variant_tag, INFERENCE_MONTH_STR,
            include_retrieval=True, output_dir=OUTPUT_DIR,
        )
        run_via_deepseek(requests_k, variant_tag, OUTPUT_DIR, api_key)

        batch_requests_by_id = {req["custom_id"]: req for req in requests_k}
        repair_incomplete_rows_deepseek(
            out_path=f"{OUTPUT_DIR}/test_pred_results_{variant_tag}.csv",
            batch_requests_by_id=batch_requests_by_id,
            is_valid_result_fn=is_valid_result,
            api_key=api_key,
        )

    df_no_retrieval = load_retrieval_csv(K_VALUES[0])
    df_no_retrieval = df_no_retrieval.drop(columns=['retrieved_context']).copy()
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, "no_retrieval", INFERENCE_MONTH_STR,
        include_retrieval=False, output_dir=OUTPUT_DIR,
    )
    run_via_deepseek(requests_no_retrieval, "no_retrieval", OUTPUT_DIR, api_key)

    batch_requests_by_id_no_retrieval = {req["custom_id"]: req for req in requests_no_retrieval}
    repair_incomplete_rows_deepseek(
        out_path=f"{OUTPUT_DIR}/test_pred_results_no_retrieval.csv",
        batch_requests_by_id=batch_requests_by_id_no_retrieval,
        is_valid_result_fn=is_valid_result,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
