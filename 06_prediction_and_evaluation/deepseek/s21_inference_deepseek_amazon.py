"""
Amazon (baby / home_and_kitchen) inference client -- DeepSeek variant.

Sibling of s21_inference_vllm_qwen_amazon.py / s21_inference_vllm_amazon.py,
but calling DeepSeek's hosted API (deepseek_client.py) instead of a
locally-served vLLM model. Reuses build_target_item_block /
PRODUCT_SYSTEM_PROMPT / load_retrieval_csv / _resolve_inference_month from
s18_prepare_for_prediction_amazon.py -- same source of truth for the
target-item prompt block as every other model backend in this pipeline.

What's different from the vLLM Amazon scripts (see deepseek_client.py's
module docstring for the full rationale on each of these):
  - No --seed / seed field at all (not supported by DeepSeek's API, and
    not needed -- a hosted API naturally varies run-to-run, same as
    GPT-4o-mini elsewhere in this study).
  - No logit_bias / emoji banning (not supported).
  - No local tokenizer / precise prompt-token budgeting -- max_tokens is a
    fixed ceiling given DeepSeek's much larger context window.
  - response_format is generic {"type": "json_object"}, not the strict
    json_schema mode vLLM uses -- expect a somewhat higher invalid-JSON
    rate, caught by the same is_valid_result()/repair-pass safety net.
  - Sequential (one request at a time) since DeepSeek has no batch API --
    functionally the same shape as run_via_vllm, just against a hosted
    endpoint with retry/backoff on transient errors.
  - Runs directly on any machine with network access (no GPU / vLLM
    server needed) -- no PBS submission required, though you can still
    wrap this in a lightweight CPU job if your cluster prefers everything
    go through the scheduler.

Run per category, e.g.:
    export DEEPSEEK_API_KEY=sk-...
    python3 s21_inference_deepseek_amazon.py --category baby
    python3 s21_inference_deepseek_amazon.py --category home_and_kitchen --output-dir deepseek_flash_output/amazon_home_and_kitchen/run_2
"""
import argparse
import json
import os

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "gpt4omini"))
from s18_prepare_for_prediction_amazon import (
    build_target_item_block,
    PRODUCT_SYSTEM_PROMPT,
    load_retrieval_csv,
    _resolve_inference_month,
)
from deepseek_client import (
    DEEPSEEK_MODEL,
    MAX_TOKENS_CEILING,
    build_deepseek_body,
    get_api_key,
    run_via_deepseek,
    repair_incomplete_rows_deepseek,
)

# ── CONFIG ────────────────────────────────────────────────────────────────
NO_RETRIEVAL_FALLBACK = "No similar past items provided."
# ── END CONFIG ────────────────────────────────────────────────────────────


def prepare_batch_input(df, variant_tag, inference_month_str, include_retrieval=True,
                         output_dir=".", description_col="product_description"):
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

        body = build_deepseek_body(PRODUCT_SYSTEM_PROMPT, user_content, max_tokens=MAX_TOKENS_CEILING)

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
    if not score_str.isdigit():
        return False
    stripped_timestamp = inference_month_str.replace("-", "")
    if score_str == stripped_timestamp:
        return False
    if len(score_str) > 5:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek batch inference (sequential -- no batch API) "
                     "for one Amazon category, from s17's retrieval-context "
                     "CSV(s), with resume + repair passes."
    )
    parser.add_argument('--category', choices=['baby', 'home_and_kitchen'], required=True)
    parser.add_argument('--api-key', default=None,
                         help="DeepSeek API key. Defaults to the DEEPSEEK_API_KEY env var.")
    parser.add_argument('--retrieval-csv-template', default=None,
                         help="Defaults to processed_results/{category}/retrieved_context/"
                              "{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv "
                              "(s17_retrieved_item_formatting_component_ablation.py's default output naming) -- "
                              "must contain a literal '{{k}}' placeholder.")
    parser.add_argument('--k-values', default="1,5,10",
                         help="Comma-separated K values, matching the ones s17 was run with.")
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to deepseek_flash_output/amazon_{category}/run_1")
    parser.add_argument('--inference-month', default=None,
                         help="Overrides the predict_timestamp read from the data.")
    parser.add_argument('--description-col', default='product_description')
    parser.add_argument('--max-retries', type=int, default=3,
                         help="Repair-pass retry attempts per invalid row.")
    parser.add_argument('--sleep-seconds', type=float, default=0.0,
                         help="Optional fixed delay between requests, in case you hit "
                              "DeepSeek's rate limits when running many rows sequentially. "
                              "Transient errors already retry with backoff regardless.")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    category = args.category
    output_dir = args.output_dir or f"deepseek_flash_output/amazon_{category}/run_1"
    k_values = [int(k) for k in args.k_values.split(",")]
    retrieval_csv_template = args.retrieval_csv_template or (
        f"processed_results/{category}/retrieved_context/"
        f"{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv"
    )

    print(f"category={category}  model={DEEPSEEK_MODEL}  output_dir={output_dir}")

    for k in k_values:
        df = load_retrieval_csv(retrieval_csv_template.format(k=k))
        inference_month_str = _resolve_inference_month(df, args.inference_month)
        variant_tag = f"{category}_k{k}_full"

        requests_k = prepare_batch_input(
            df, variant_tag, inference_month_str,
            include_retrieval=True, output_dir=output_dir, description_col=args.description_col,
        )
        run_via_deepseek(requests_k, variant_tag, output_dir, api_key, sleep_seconds=args.sleep_seconds)

        batch_requests_by_id = {req["custom_id"]: req for req in requests_k}
        repair_incomplete_rows_deepseek(
            out_path=f"{output_dir}/test_pred_results_{variant_tag}.csv",
            batch_requests_by_id=batch_requests_by_id,
            is_valid_result_fn=lambda c: is_valid_result(c, inference_month_str),
            api_key=api_key,
            max_retries=args.max_retries,
        )

    df_no_retrieval = load_retrieval_csv(retrieval_csv_template.format(k=k_values[0]))
    inference_month_str = _resolve_inference_month(df_no_retrieval, args.inference_month)
    df_no_retrieval = df_no_retrieval.drop(columns=['retrieved_context']).copy()
    variant_tag_nr = f"{category}_no_retrieval"
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, variant_tag_nr, inference_month_str,
        include_retrieval=False, output_dir=output_dir, description_col=args.description_col,
    )
    run_via_deepseek(requests_no_retrieval, variant_tag_nr, output_dir, api_key, sleep_seconds=args.sleep_seconds)

    batch_requests_by_id_no_retrieval = {req["custom_id"]: req for req in requests_no_retrieval}
    repair_incomplete_rows_deepseek(
        out_path=f"{output_dir}/test_pred_results_{variant_tag_nr}.csv",
        batch_requests_by_id=batch_requests_by_id_no_retrieval,
        is_valid_result_fn=lambda c: is_valid_result(c, inference_month_str),
        api_key=api_key,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
