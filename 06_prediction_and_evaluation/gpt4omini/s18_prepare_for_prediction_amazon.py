"""
Amazon (baby / home_and_kitchen) adaptation of
s18_fixed_prepare_for_prediction_v4_2_neutral_test.py.

Prediction batch prep: builds the final LLM prediction request (system
prompt + target item block + retrieved-context block) from
s17_retrieved_item_formatting_component_ablation.py's per-query retrieval-context CSV.

What's UNCHANGED: the overall batch-construction shape (prepare_batch_input/
save_jsonl), _month_label/_month_before date-window helpers, and
build_target_trend_line's rendering logic (reused directly -- it already
takes trend_array/release_date/inference_month_str as generic parameters).

What changed, and why:

  1. No movie metadata. The movie reference's build_movie_description()
     assembled a templated paragraph from ~9 structured columns (genre,
     director, writers, top5_actors, producers, music, language,
     release_date, synopsis) that simply don't exist in this Amazon
     pipeline -- s5_preprocess_datasets_full.py's output only ever carries a
     single free-text column (product_description, or item_description on
     the training side -- see build_item_description's column-name
     fallback). So build_item_description() below just returns that text
     directly instead of templating it into a sentence -- there's nothing
     structured to template. capitalize_names/_fallback_na (used only for
     the removed cast/crew fields) and sample_equal_per_direction (a
     movie-dataset-specific stratified sampler, unused in main() even in
     the reference) are dropped as dead weight.

  2. trend_array is re-parsed with float(), not int() -- matching every
     other _amazon script in this pipeline (_TREND_ARRAY_SPLIT_PATTERN's
     int(...) cast in load_retrieval_csv is replaced).

  3. The system prompt is reworded for a generic e-commerce product
     instead of a movie ("MovieDescription"/"movie popularity analyst" ->
     "ProductDescription"/"product popularity analyst"), but keeps the
     exact same two-step reasoning contract and JSON output shape as the
     movie reference's "Simple Prompt V5" (predict from
     ProductDescription+PopularityHistory first, then adjust using
     SimilarPastItems) -- see PRODUCT_SYSTEM_PROMPT below. The many other
     prompt variants commented out in the movie reference (v2, v4, the
     LatestKnownScore-anchored version) are not carried over -- only the
     one actually wired into prepare_batch_input's call in main().

  4. INFERENCE_MONTH_STR is read from the retrieval-context CSV's own
     predict_timestamp column (s5_preprocess_datasets_full.py stamps every
     row with the same fixed month, e.g. "2014-06") rather than hardcoded,
     with an explicit mismatch check in case a future test set ever mixes
     months -- --inference-month overrides this if you need to force one.

  5. main() is parameterized by --category baby|home_and_kitchen, with Amazon
     file-naming defaults matching s17's actual output filenames
     (RETRIEVAL_CSV_TEMPLATE below), instead of the movie project's fixed
     processed_results/260828_test/... paths.

  6. normalize_explanatory_factor_bullets / _BULLET_FIX_PATTERN are kept
     UNCHANGED -- they clean up the same "[Domain - Category - Impact]:"
     bullet shape amazon_system_prompt.txt's ANSWER_FORMAT produces (see
     s17_retrieved_item_formatting_component_ablation.py's docstring point 1), so
     they're directly reusable once explanatory_factors is populated,
     even though nothing calls them yet in this script's main() (same as
     in the movie reference).

Run per category, e.g.:
    python s18_prepare_for_prediction_amazon.py --category baby
"""
import argparse
import json
import os
import re
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
_BULLET_FIX_PATTERN = re.compile(
    r"\[(Lifecycle|Metadata)\s*[-:]\s*(.+?)\s*[-:]\s*(Positive|Negative|Neutral)\]:+"
)
_TREND_ARRAY_SPLIT_PATTERN = re.compile(r"[,\s]+")

DEFAULT_MODEL = 'gpt-4o-mini-2024-07-18'
NO_RETRIEVAL_FALLBACK = "No similar past items provided."
# ── END CONFIG ────────────────────────────────────────────────────────────


def normalize_explanatory_factor_bullets(text):
    """Normalize any '[Domain <sep> Category <sep> Impact]:<:>' variant to the
    canonical '[Domain - Category - Impact]:'. Idempotent: the canonical form
    matches too (dash is one of the accepted separators) but replaces with itself."""
    if not isinstance(text, str):
        return text
    return _BULLET_FIX_PATTERN.sub(r"[\1 - \2 - \3]:", text)


def _parse_trend_array_str(x):
    """Re-parses trend_array from its saved-CSV string form. Uses float()
    (not the movie reference's int()) -- Amazon values aren't guaranteed to
    print as bare integers."""
    if not isinstance(x, str):
        return x
    return np.array([int(i) for i in _TREND_ARRAY_SPLIT_PATTERN.split(x.strip("[]").strip()) if i])


def load_retrieval_csv(path):
    df = pd.read_csv(path)
    df["trend_array"] = df["trend_array"].apply(_parse_trend_array_str)
    return df


# ── item description + trend formatting ─────────────────────────────────

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


def build_item_description(row, description_col="product_description"):
    """Amazon has no structured movie-style metadata (genre/director/cast/
    etc) to template into a sentence -- s5_preprocess_datasets_full.py's
    output carries a single free-text description column. Falls back to
    'item_description' (the training-side column name -- see
    s5_save_eight_groups.py's module docstring point 2) if
    description_col isn't present, since a retrieval-context CSV built
    from training-side data would use that name instead."""
    if description_col in row and pd.notna(row[description_col]):
        return _fallback_na(row[description_col])
    if "item_description" in row and pd.notna(row["item_description"]):
        return _fallback_na(row["item_description"])
    return "NA"


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
    """Same rendering as the retrieved-item cards' Trend line (see
    s17_retrieved_item_formatting_component_ablation.py's _format_trend_for_card) --
    matching the retrieved items' format here matters: giving the target a
    differently-shaped history than its comps risks the model treating
    them as two different kinds of object instead of the same kind of
    sequence.

    Returns None for a debut item (empty or [-1]-sentinel trend_array) --
    callers should render an explicit "No historical trend" note instead.
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


def build_target_item_block(row, inference_month_str, description_col="product_description"):
    item_description_text = build_item_description(row, description_col=description_col)

    if row["item_age"] == -1:
        # Debut item -- no release history yet, same convention as the
        # retrieved-item debut cards (which omit the Trend line entirely).
        popularity_history_text = "Debut item -- no historical trend available"
    else:
        trend_line = build_target_trend_line(
            row["trend_array"], row.get("release_month"), inference_month_str
        )
        popularity_history_text = trend_line if trend_line is not None else "No historical trend"

    return f"""
<ProductDescription>{item_description_text}</ProductDescription>

<PopularityHistory>{popularity_history_text}</PopularityHistory>

<InferenceTimestamp>{inference_month_str}</InferenceTimestamp>
"""


# ── system prompt (reworded "Simple Prompt V5" -- see module docstring point 3) ──

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
  given <InferenceTimestamp>. Non-negative integer, formatted as a string.

Output only this JSON object, with nothing before or after.

### Input format:

<ProductDescription>...</ProductDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
<SimilarPastItems>...</SimilarPastItems>
"""


def prepare_batch_input(df, model, variant_tag, inference_month_str,
                         include_retrieval=True, output_dir=".",
                         description_col="product_description"):
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

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": 4096,
            "temperature": 0.1,
            "top_p": 1.0,
            "stream": False,
        }

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        batch_requests.append(request)

    mapping_df = df.copy()
    mapping_df["custom_id"] = custom_ids
    if not include_retrieval and "retrieved_context" in mapping_df.columns:
        mapping_df = mapping_df.drop(columns=["retrieved_context"])

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = f"{output_dir}/mapping_{variant_tag}.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping ({len(mapping_df)} rows) -> {mapping_path}")
    print(f"Created JSONL batch with {len(batch_requests)} requests")
    return batch_requests


def save_jsonl(requests, variant_tag, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/pred_batch_{variant_tag}.jsonl"
    with open(output_file, "w") as f:
        for item in requests:
            f.write(json.dumps(item) + "\n")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(requests)} requests -> {output_file} ({size_mb:.1f} MB)")
    return output_file


def _resolve_inference_month(df, override):
    """Reads INFERENCE_MONTH_STR from the retrieval-context CSV's own
    predict_timestamp column (stamped by s5_preprocess_datasets_full.py) if
    present, rather than hardcoding it like the movie reference did --
    with a mismatch check in case a future test set ever mixes months.
    --inference-month always wins if given."""
    if override:
        return override
    if "predict_timestamp" in df.columns:
        values = df["predict_timestamp"].dropna().unique()
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            raise ValueError(
                f"predict_timestamp has multiple distinct values in this data "
                f"({sorted(values)}) -- pass --inference-month explicitly to pick one."
            )
    raise ValueError(
        "Could not determine the inference month -- no 'predict_timestamp' column "
        "found and --inference-month wasn't given."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the final prediction LLM batch input for one "
                     "Amazon category, from s17's retrieval-context CSV(s)."
    )
    parser.add_argument('--category', choices=['baby', 'home_and_kitchen'], required=True)
    parser.add_argument('--retrieval-csv-template', default=None,
                         help="Defaults to processed_results/{category}/retrieved_context/"
                              "{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv "
                              "(s17_retrieved_item_formatting_component_ablation.py's default output naming) -- "
                              "must contain a literal '{{k}}' placeholder.")
    parser.add_argument('--k-values', default="1,5,10",
                         help="Comma-separated K values, matching the ones s17 was run with.")
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to prediction_results/{category}")
    parser.add_argument('--inference-month', default=None,
                         help="Overrides the predict_timestamp read from the data (see "
                              "_resolve_inference_month).")
    parser.add_argument('--description-col', default='product_description',
                         help="Column holding the item's free-text description. "
                              "s5_preprocess_datasets_full.py's output uses "
                              "'product_description'; falls back to 'item_description' "
                              "if this column isn't present on a given row.")
    args = parser.parse_args()

    category = args.category
    output_dir = args.output_dir or f"prediction_results/{category}"
    k_values = [int(k) for k in args.k_values.split(",")]
    retrieval_csv_template = args.retrieval_csv_template or (
        f"processed_results/{category}/retrieved_context/"
        f"{category}_retrieval_context_per_query_embed_rerank_k{{k}}_a50_b50_gtprior0.csv"
    )

    for k in k_values:
        df = load_retrieval_csv(retrieval_csv_template.format(k=k))
        inference_month_str = _resolve_inference_month(df, args.inference_month)
        variant_tag = f"{category}_k{k}_a50_b50_gtprior0"
        requests = prepare_batch_input(
            df, args.model, variant_tag, inference_month_str,
            include_retrieval=True, output_dir=output_dir,
            description_col=args.description_col,
        )
        save_jsonl(requests, variant_tag, output_dir)

    df_no_retrieval = load_retrieval_csv(retrieval_csv_template.format(k=k_values[0]))
    inference_month_str = _resolve_inference_month(df_no_retrieval, args.inference_month)
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, args.model, f"{category}_no_retrieval", inference_month_str,
        include_retrieval=False, output_dir=output_dir,
        description_col=args.description_col,
    )
    save_jsonl(requests_no_retrieval, f"{category}_no_retrieval", output_dir)


if __name__ == "__main__":
    main()
