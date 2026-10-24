"""
Repair pass for GPT-4o-mini batch prediction results -- reusable across
Douban, Amazon Baby, and Home & Kitchen (and any other dataset/variant),
since only the file paths and inference month change between runs.

Applies the SAME validation gate as the vLLM/DeepSeek repair scripts'
is_valid_result(), adapted for GPT-4o-mini's already-parsed
predict_popularity_score/reasoning columns instead of a raw JSON
`content` column (no JSON-parse step needed on the existing row; only
on a freshly retried response).

Usage:
    export OPENAI_API_KEY=...
    python s21_repair_gpt4o_mini.py \
        --results-csv gpt4omini_output/home_and_kitchen/api_call_results/run_1/0925_kitchen_test_no_retrieval.csv \
        --original-jsonl gpt4omini_output/home_and_kitchen/pred_batch_home_and_kitchen_no_retrieval.jsonl \
        --inference-month 2019-11
"""
import argparse
import json
import os

import pandas as pd
from openai import OpenAI


def is_valid_result(score, reasoning, inference_month_str: str) -> bool:
    """Mirrors is_valid_result() in the vLLM/DeepSeek repair scripts exactly:
    - reasoning must be a non-empty string of at least 30 characters
    - score must be present and digits-only
    - score must not exactly equal the inference month written as YYYYMM
      (guards against the model echoing the prediction-month timestamp)
    - score must be at most 5 digits (your real popularity values never
      reach that scale, so this doubles as a plausibility check)
    """
    if not reasoning or not isinstance(reasoning, str) or len(reasoning.strip()) < 30:
        return False
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return False
    score_str = str(score).strip()
    if score_str.endswith(".0"):
        score_str = score_str[:-2]
    if not score_str.isdigit():
        return False
    stripped_timestamp = inference_month_str.replace("-", "")
    if score_str == stripped_timestamp:
        return False
    if len(score_str) > 5:
        return False
    return True


def repair_gpt4o_mini_results(results_csv: str, original_jsonl: str, inference_month_str: str,
                               api_key: str, max_retries: int = 3, model_override: str = None):
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(results_csv)
    needs_repair_mask = ~df.apply(
        lambda row: is_valid_result(row.get("predict_popularity_score"), row.get("reasoning"), inference_month_str),
        axis=1,
    )
    missing_ids = set(df.loc[needs_repair_mask, "custom_id"])
    print(f"{results_csv}: {len(missing_ids)} / {len(df)} rows need repair")

    if not missing_ids:
        return df

    requests_by_id = {}
    with open(original_jsonl) as f:
        for line in f:
            req = json.loads(line)
            if req["custom_id"] in missing_ids:
                requests_by_id[req["custom_id"]] = req

    df = df.set_index("custom_id")
    if "was_repaired" not in df.columns:
        df["was_repaired"] = False
        df["repair_temperature"] = None

    for custom_id in missing_ids:
        if custom_id not in requests_by_id:
            print(f"  skip {custom_id}: no original request found in {original_jsonl}")
            continue

        req = requests_by_id[custom_id]
        body = req["body"]
        fixed = False

        for attempt in range(1, max_retries + 1):
            temp = min(0.2 + 0.2 * attempt, 0.8)  # same schedule as the vLLM/DeepSeek scripts: 0.4, 0.6, 0.8
            try:
                resp = client.chat.completions.create(
                    model=model_override or body["model"],
                    messages=body["messages"],
                    temperature=temp,
                    response_format=body.get("response_format"),
                )
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                score = parsed.get("predict_popularity_score")
                reasoning = parsed.get("reasoning")
                if is_valid_result(score, reasoning, inference_month_str):
                    df.loc[custom_id, "predict_popularity_score"] = float(score)
                    df.loc[custom_id, "reasoning"] = reasoning
                    df.loc[custom_id, "was_repaired"] = True
                    df.loc[custom_id, "repair_temperature"] = temp
                    print(f"  {custom_id} fixed at temp={temp} (attempt {attempt})")
                    fixed = True
                    break
                else:
                    print(f"  {custom_id} attempt {attempt} still invalid")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  {custom_id} attempt {attempt} unparseable response: {e}")
            except Exception as e:
                print(f"  {custom_id} attempt {attempt} errored: {e}")

        if not fixed:
            # attempted but not fixed -- original (first-call) content is kept untouched,
            # same "keep original, not last retry" behavior as the vLLM/DeepSeek scripts
            df.loc[custom_id, "was_repaired"] = True
            print(f"  {custom_id} still bad after {max_retries} attempts -- original kept for manual review")

    df = df.reset_index()
    df.to_csv(results_csv, index=False)

    still_bad = (~df.apply(
        lambda row: is_valid_result(row.get("predict_popularity_score"), row.get("reasoning"), inference_month_str),
        axis=1,
    )).sum()
    print(f"Done. {still_bad} rows still invalid in {results_csv}.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repair invalid/missing GPT-4o-mini predictions. Reusable across "
                     "Douban, Amazon Baby, and Home & Kitchen -- only --results-csv, "
                     "--original-jsonl, and --inference-month change between runs."
    )
    parser.add_argument("--results-csv", required=True,
                         help="CSV with custom_id, predict_popularity_score, reasoning columns "
                              "(overwritten in place with repaired values).")
    parser.add_argument("--original-jsonl", required=True,
                         help="The original pred_batch_*.jsonl this results CSV was generated from.")
    parser.add_argument("--inference-month", required=True,
                         help="Prediction month as YYYY-MM, e.g. 2019-11 -- used only to guard "
                              "against the model echoing this as its score.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--model", default=None,
                         help="Override the model name from the original request body, if needed.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the OPENAI_API_KEY environment variable before running this script.")

    repair_gpt4o_mini_results(
        results_csv=args.results_csv,
        original_jsonl=args.original_jsonl,
        inference_month_str=args.inference_month,
        api_key=api_key,
        max_retries=args.max_retries,
        model_override=args.model,
    )