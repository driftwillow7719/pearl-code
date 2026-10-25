"""
Amazon (baby / home_and_kitchen) adaptation of the Llama-3-8B variant

Run per category, e.g.:
    python3 s21_inference_vllm_llama3_amazon.py --category baby
    python3 s21_inference_vllm_llama3_amazon.py --category home_and_kitchen --model llama-3-8b
"""
import argparse
import csv
import json
import os
import pandas as pd
import requests as http
from transformers import AutoTokenizer

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "gpt4omini"))
from s18_prepare_for_prediction_amazon import (
    build_target_item_block,
    # PRODUCT_SYSTEM_PROMPT,
    load_retrieval_csv,
    _resolve_inference_month,
)
# ── CONFIG ────────────────────────────────────────────────────────────────
PRODUCT_SYSTEM_PROMPT = """You are an expert e-commerce product popularity analyst. Your task is to predict a product's popularity score at a given point in time, based on its own
descriptive features, its historical popularity trajectory (when available), and a set of retrieved past products with similar description and popularity patterns.

### About <PopularityHistory>:
<PopularityHistory> covers the target's own monthly popularity values from
release through one month before <InferenceTimestamp> -- it does not
include the value at <InferenceTimestamp> itself, which is what you are
being asked to predict.

### About <SimilarPastItems>:
<SimilarPastItems> shows past products retrieved for similar description and
popularity pattern -- they will not necessarily match the target's outcome.
For non-debut items, each item's own trend line covers its own history up through one month before its own outcome month; its outcome is stated only as a relative percentage change from that trend's last value (never the raw outcome number itself), showing how much the observed popularity moved.
For debut items, neither a trend line nor an outcome value is provided, since there is no prior popularity history -- only its own product description and explanatory factors are available for these cases.

Steps:
1. First, estimate the target's popularity score using only <ProductDescription>
   and <PopularityHistory> (when available). It's a specific number, not a category label.
2. Then review <SimilarPastItems>. For each retrieved item, check whether
   its trend trajectory and explanatory factors -- covering both its own
   product description and its own popularity history -- are actually
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

<ProductDescription>...</ProductDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
<SimilarPastItems>...</SimilarPastItems>
"""

NO_RETRIEVAL_FALLBACK = "No similar past items provided."

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

# Llama-3-8B's own native (non-extended) context window is 8192 -- this
# matches Llama-2's budget without needing to cap anything down (unlike
# Qwen2, which natively supports 32768 and is capped here to compare fairly).
MAX_MODEL_LEN = 8192
SAFETY_MARGIN = 50
MAX_TOKENS_CEILING = 1600
# ── END CONFIG ────────────────────────────────────────────────────────────


def build_emoji_logit_bias(tokenizer, bias_value=-100):
    """Ban the byte-fallback tokens that start a 4-byte UTF-8 sequence
    (lead bytes 0xF0-0xF4). NOTE: this was written for Llama-2's
    SentencePiece tokenizer, which represents unseen Unicode as explicit
    '<0xNN>' byte tokens. Llama-3 uses a different (tiktoken-style BPE,
    128k-vocab) tokenizer that may not expose the same '<0xF0>'-style token
    strings -- if none of these are found in Llama-3's vocab, `bias` will
    come back empty and this mechanism will do nothing (silently, not an
    error). See the startup warning in main()."""
    vocab = tokenizer.get_vocab()
    bias = {}
    for lead_byte in ["0xF0", "0xF1", "0xF2", "0xF3", "0xF4"]:
        token_str = f"<{lead_byte}>"
        if token_str in vocab:
            bias[str(vocab[token_str])] = bias_value
    return bias


def prepare_batch_input(df, model, variant_tag, inference_month_str, tokenizer,
                         emoji_logit_bias, include_retrieval=True, output_dir=".",
                         description_col="product_description", seed=None):
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

        target_item_block = build_target_item_block(row, inference_month_str, description_col=description_col)

        user_content = f"""{target_item_block}

        {retrieved_section}
        """

        prompt_tokens = len(tokenizer.encode(PRODUCT_SYSTEM_PROMPT + user_content))
        max_tokens = min(MAX_TOKENS_CEILING, MAX_MODEL_LEN - prompt_tokens - SAFETY_MARGIN)
        if max_tokens < 150:
            print(f"WARNING: {custom_id} has very little completion room ({max_tokens} tokens, prompt={prompt_tokens}) — may truncate or fail")
            max_tokens = max(max_tokens, 1)

        print(f"{custom_id}: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "top_p": 1,
            "logit_bias": emoji_logit_bias,
            "repetition_penalty": 1.15,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "product_popularity_prediction",
                    "schema": OUTPUT_JSON_SCHEMA,
                },
            },
        }

        # Per-request RNG seed. vLLM's OpenAI-compatible server otherwise
        # defaults to the same RNG state across independent `vllm serve`
        # launches, which produces byte-identical output (incl. free-text
        # reasoning) across nominally-independent replicate runs even with
        # temperature > 0 -- temperature/top_p shape the sampling
        # distribution, but seed determines which draw you get from it.
        # Pass a DIFFERENT --seed per run (1/2/3, ...) to get genuine
        # run-to-run variance; the same seed on every run instead gives you
        # reproducible-but-still-identical output, same as leaving it unset.
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


def run_via_vllm(batch_requests, variant_tag, output_dir=".", host="http://localhost:8000"):
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


# def is_valid_result(content: str, inference_month_str: str) -> bool:
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

def is_valid_result(content: str, inference_month_str: str) -> bool:
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
    inference_month_str: str,
    host: str,
    max_retries: int = 3,
):
    """Reread out_path, find rows that failed to produce valid JSON,
    resend just those requests, and rewrite the CSV in place.

    Every row gets a `was_repaired` (bool) and `repair_temperature` column
    written -- whether the repair attempt succeeded or not -- so there's a
    record of which rows needed a retry and at what temperature the fix
    landed, instead of silently overwriting content with no trace a repair
    happened (matches the Qwen Amazon script's repair pass)."""
    df = pd.read_csv(out_path)

    if "was_repaired" not in df.columns:
        df["was_repaired"] = False
    if "repair_temperature" not in df.columns:
        df["repair_temperature"] = None

    bad_mask = ~df["content"].apply(lambda c: is_valid_result(c, inference_month_str))
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

            if is_valid_result(content, inference_month_str):
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
    still_bad = (~df["content"].apply(lambda c: is_valid_result(c, inference_month_str))).sum()
    print(f"Repair pass done. {still_bad} rows remain invalid.")


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM batch inference (Llama-3-8B) for one Amazon "
                     "category, from s17's retrieval-context CSV(s), with "
                     "resume + repair passes."
    )
    parser.add_argument('--category', choices=['baby', 'home_and_kitchen'], required=True)
    parser.add_argument('--model', default="llama-3-8b",
                         help="Must match the --model/positional arg vLLM was started with.")
    parser.add_argument('--tokenizer-path', default=None,
                         help="Defaults to --model (local HF-format checkpoint dir/name).")
    parser.add_argument('--vllm-host', default=None,
                         help="Overrides --vllm-port / VLLM_PORT entirely, e.g. http://localhost:8001.")
    parser.add_argument('--vllm-port', default=os.environ.get("VLLM_PORT", "8000"),
                         help="Used to build http://localhost:<port> when --vllm-host isn't given. "
                              "Defaults to the VLLM_PORT env var (as exported by the PBS job script), else 8000.")
    parser.add_argument('--retrieval-csv-template', default=None,
                         help="Defaults to processed_results/{category}/retrieved_context/"
                              "{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv "
                              "(s17_retrieved_item_formatting_component_ablation.py's default output naming) -- "
                              "must contain a literal '{{k}}' placeholder.")
    parser.add_argument('--k-values', default="1,5,10",
                         help="Comma-separated K values, matching the ones s17 was run with.")
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to llama_3_8b_output/amazon_{category}/run_1")
    parser.add_argument('--inference-month', default=None,
                         help="Overrides the predict_timestamp read from the data.")
    parser.add_argument('--description-col', default='product_description')
    parser.add_argument('--max-retries', type=int, default=3,
                         help="Repair-pass retry attempts per invalid row.")
    parser.add_argument('--seed', type=int, default=None,
                         help="Per-request RNG seed passed to vLLM (e.g. 1/2/3 for "
                              "replicate runs 1/2/3). Left unset, vLLM's OpenAI-"
                              "compatible server defaults to the same RNG state on "
                              "every launch, so 'independent' runs come back "
                              "byte-identical despite temperature=0.1 -- pass a "
                              "DIFFERENT --seed per run to get genuine variance.")
    args = parser.parse_args()

    category = args.category
    vllm_host = args.vllm_host or f"http://localhost:{args.vllm_port}"
    output_dir = args.output_dir or f"llama_3_8b_output/amazon_{category}/run_1"
    k_values = [int(k) for k in args.k_values.split(",")]
    retrieval_csv_template = args.retrieval_csv_template or (
        f"processed_results/{category}/retrieved_context/"
        f"{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv"
    )

    if args.seed is None:
        print("WARNING: --seed not set -- vLLM will use its default RNG state, "
              "which is typically the SAME across independent server launches. "
              "If you're running multiple replicates to measure run-to-run "
              "variance, pass a different --seed per run (e.g. 1, 2, 3) or every "
              "run will come back identical.")

    print(f"category={category}  model={args.model}  vllm_host={vllm_host}  output_dir={output_dir}  seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model)
    emoji_logit_bias = build_emoji_logit_bias(tokenizer)
    if not emoji_logit_bias:
        print("WARNING: emoji_logit_bias is empty -- Llama-3's tokenizer doesn't "
              "expose '<0xF0>'-style byte tokens the way Llama-2's does. Emoji "
              "banning via logit_bias will have NO effect for this model. If "
              "emoji/symbol issues show up in Llama-3 outputs, this needs a "
              "different approach (e.g. banning specific emoji token ids "
              "directly, found by tokenizing sample emoji and inspecting ids).")

    for k in k_values:
        df = load_retrieval_csv(retrieval_csv_template.format(k=k))
        inference_month_str = _resolve_inference_month(df, args.inference_month)
        variant_tag = f"{category}_k{k}_full"

        requests_k = prepare_batch_input(
            df, args.model, variant_tag, inference_month_str, tokenizer, emoji_logit_bias,
            include_retrieval=True, output_dir=output_dir, description_col=args.description_col,
            seed=args.seed,
        )
        run_via_vllm(requests_k, variant_tag, output_dir, host=vllm_host)

        batch_requests_by_id = {req["custom_id"]: req for req in requests_k}
        repair_incomplete_rows(
            out_path=f"{output_dir}/test_pred_results_{variant_tag}.csv",
            batch_requests_by_id=batch_requests_by_id,
            inference_month_str=inference_month_str,
            host=vllm_host,
            max_retries=args.max_retries,
        )

    df_no_retrieval = load_retrieval_csv(retrieval_csv_template.format(k=k_values[0]))
    inference_month_str = _resolve_inference_month(df_no_retrieval, args.inference_month)
    df_no_retrieval = df_no_retrieval.drop(columns=['retrieved_context']).copy()
    variant_tag_nr = f"{category}_no_retrieval"
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, args.model, variant_tag_nr, inference_month_str, tokenizer, emoji_logit_bias,
        include_retrieval=False, output_dir=output_dir, description_col=args.description_col,
        seed=args.seed,
    )
    run_via_vllm(requests_no_retrieval, variant_tag_nr, output_dir, host=vllm_host)

    batch_requests_by_id_no_retrieval = {req["custom_id"]: req for req in requests_no_retrieval}
    repair_incomplete_rows(
        out_path=f"{output_dir}/test_pred_results_{variant_tag_nr}.csv",
        batch_requests_by_id=batch_requests_by_id_no_retrieval,
        inference_month_str=inference_month_str,
        host=vllm_host,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
