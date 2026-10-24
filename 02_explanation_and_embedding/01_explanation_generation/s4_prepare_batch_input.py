"""
Builds the OpenAI Batch API input (.jsonl) used to generate explanations for
the training data. Works for every dataset; the only dataset-specific parts
are how the item metadata block is written and how "months of history" is
computed (see DATASET_FORMATS below).

Run:
    python s4_prepare_batch_input.py --dataset douban \
        --csv <synthetic training csv> --prompt <system prompt .txt> --out-dir <dir>
"""
import argparse
import json
import os

import pandas as pd


def month_diff(month_from: str, month_to: str) -> int:
    """Number of months from `month_from` to `month_to` ('YYYY-MM' strings)."""
    y1, m1 = map(int, str(month_from).split('-'))
    y2, m2 = map(int, str(month_to).split('-'))
    return (y2 - y1) * 12 + (m2 - m1)


# ---------------------------------------------------------------------------
# dataset-specific pieces
# ---------------------------------------------------------------------------

def _douban_metadata_block(row):
    return f"""<MOVIE_METADATA>
movie_name: {row['movie_name']}
release_date: {row['release_date']}
genre: {row['genre']}
description: {row['description']}
language: {row['language']}
director: {row['director']}
writer: {row['writers']}
top5_actors: {row['top5_actors']}
producers: {row['producers']}
</MOVIE_METADATA>"""


def _douban_months_of_history(row):
    return row['item_age'] + 1


def _amazon_metadata_block(row, tag="PRODUCT_METADATA"):
    return f"""<{tag}>
item_description: {row['item_description']}
release_month: {row['release_month']}
</{tag}>"""


def _amazon_months_of_history(row):
    return month_diff(row['release_month'], row['popularity_month'])


DATASET_FORMATS = {
    "douban":           dict(metadata=_douban_metadata_block, months=_douban_months_of_history),
    "baby":             dict(metadata=_amazon_metadata_block, months=_amazon_months_of_history),
    "home_and_kitchen": dict(metadata=_amazon_metadata_block, months=_amazon_months_of_history),
}


# ---------------------------------------------------------------------------
# shared logic
# ---------------------------------------------------------------------------

def prepare_batch_input_from_csv(dataset: str,
                                 prompt_path: str,
                                 csv_file_path: str,
                                 output_jsonl_path: str,
                                 mapping_output_path: str = None,
                                 model: str = "gpt-5.6-luna"):
    fmt = DATASET_FORMATS[dataset]

    df = pd.read_csv(csv_file_path)
    print(f"Read CSV file: {csv_file_path}")
    print(f"Total data: {len(df)}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        BASE_PROMPT = f.read().strip()

    batch_requests = []
    custom_ids = []

    for idx, row in df.iterrows():
        custom_id = f"request-{idx+1:05d}"   # strictly incrementing, no collisions
        custom_ids.append(custom_id)

        system_prompt = BASE_PROMPT
        temperature = 0.1
        reasoning_effort = "none"

        has_history = (
            pd.notna(row['historical_trend'])
            and str(row['historical_trend']).strip() != ''
            and str(row['historical_trend']).strip() != 'No historical trend'
        )

        months = fmt['months'](row)
        if has_history:
            historical_trend_block = f"""months_of_history_available: {months}
popularity_trajectory: {row['historical_trend']}
latest_pop_value: {row['latest_pop_value']}
latest_momentum: {row['latest_momentum']}"""
        else:
            historical_trend_block = f"""months_of_history_available: {months}
popularity_trajectory: {row['historical_trend']}"""

        user_content = f"""
{fmt['metadata'](row)}

<HISTORICAL_TREND>
{historical_trend_block}
</HISTORICAL_TREND>

<PREDICTION_TARGET>
popularity_month: {row['popularity_month']}
popularity_score: {row['popularity_value']}
</PREDICTION_TARGET>
"""

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_completion_tokens": 5000,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                'stream': False
            }
        }
        batch_requests.append(request)

    with open(output_jsonl_path, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    print(f"Created JSONL file with {len(batch_requests)} requests at {output_jsonl_path}")

    # mapping file keyed by the same custom_id -- safe for merging results back
    # later regardless of how the JSONL gets split into shards
    if mapping_output_path:
        mapping_df = df.copy()
        mapping_df["custom_id"] = custom_ids
        mapping_df.to_csv(mapping_output_path, index=False)
        print(f"Saved mapping ({len(mapping_df)} rows) -> {mapping_output_path}")

    return output_jsonl_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASET_FORMATS), required=True)
    parser.add_argument('--csv', required=True, help="synthetic training data CSV")
    parser.add_argument('--prompt', required=True, help="system prompt .txt")
    parser.add_argument('--out-dir', default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prepare_batch_input_from_csv(
        dataset=args.dataset,
        prompt_path=args.prompt,
        csv_file_path=args.csv,
        output_jsonl_path=f"{args.out_dir}/{args.dataset}_full_data.jsonl",
        mapping_output_path=f"{args.out_dir}/{args.dataset}_mapping_full_data.csv",
    )


if __name__ == "__main__":
    main()