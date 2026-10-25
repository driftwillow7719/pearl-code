"""
Synthetic training rows and test set for the Amazon datasets (Baby, Home_and_Kitchen),
generated from each dataset's *_inter_train.csv and metadata file.

Per item, one target month is picked for each history-depth bucket (release month,
single-month history, 1-3, 4-6, 7-12, 13-24 and >24 months of history) and turned into a
training row: item description, historical trend, latest popularity value and momentum.
Items are never dropped for missing metadata; missing fields are rendered as "".

Inputs:  inter_month_count (monthly popularity counts per item) and begin_month (release index).
max_idx (the last month index) is derived from the data; ANCHOR_MONTH maps it to a calendar month for Baby and Home_and_Kitchen.
HOLDOUT = 2: the last two month indices are excluded from the training pool.
build_test_set(): one row per item with the last index as target.
"""

import ast
import random

import numpy as np
import pandas as pd
import json

# ============================================
# ITEM DESCRIPTION FORMATTING
# ============================================


def parse_category(categories_raw):
    """"[['Baby', 'Feeding', ...]]" -> "Baby, Feeding, ..." (first path only)."""
    if pd.isna(categories_raw):
        return ""
    try:
        paths = ast.literal_eval(categories_raw)
        return ", ".join(paths[0]) if paths else ""
    except (ValueError, SyntaxError):
        return ""


def parse_sales_rank(salesrank_raw):
    """"{'Baby': 10094}" -> "rank 10094 in Baby category"; blank/empty dict -> ""."""
    if pd.isna(salesrank_raw):
        return ""
    try:
        d = ast.literal_eval(salesrank_raw)
    except (ValueError, SyntaxError):
        return ""
    if not d:
        return ""
    key, value = next(iter(d.items()))
    return f"rank {value} in {key} category"


def format_price(price):
    return "" if pd.isna(price) else str(price)


def format_item_description(domain, title, description, brand, salesrank_raw, categories_raw, price):
    title_s = "" if pd.isna(title) else str(title).replace("\n", " ")
    desc_s = "" if pd.isna(description) else str(description).replace("\n", " ")
    brand_s = "" if pd.isna(brand) else str(brand).replace("\n", " ")
    return (
        f"The {domain} product name is {title_s}; {desc_s}; "
        f"The brand of this {domain} product is {brand_s}; "
        f"The sales rank of the product is {parse_sales_rank(salesrank_raw)}; "
        f"The category of the product are {parse_category(categories_raw)}; "
        f"The price of the item is {format_price(price)}."
    )


# ============================================
# METADATA MISSINGNESS REPORT (diagnostic only)
# ============================================

REPORT_FIELDS = ["title", "description", "categories"]


def _is_blank_string_field(val):
    """NaN, or empty/whitespace-only after stripping -> True (missing)."""
    if pd.isna(val):
        return True
    return str(val).strip() == ""


def meta_row_report_flags(row):
    """Present/missing flags for title, description and categories (diagnostic only)."""
    return {
        "title": not _is_blank_string_field(row.get("title")),
        "description": not _is_blank_string_field(row.get("description")),
        "categories": bool(parse_category(row.get("categories"))),
    }


def build_descriptions_from_meta(meta_path, domain):
    """item_id -> formatted item_description for every item in the meta file.

    Returns (descriptions_dict, n_total_in_meta, missingness_report).
    """
    meta = pd.read_csv(meta_path)
    n_total = len(meta)

    flags = meta.apply(meta_row_report_flags, axis=1, result_type="expand")
    missingness_report = {field: int((~flags[field]).sum()) for field in REPORT_FIELDS}

    descriptions = {}
    for _, row in meta.iterrows():
        descriptions[row["item_id"]] = format_item_description(
            domain, row["title"], row["description"], row["brand"], row["salesRank"],
            row["categories"], row["price"],
        )
    return descriptions, n_total, missingness_report


random.seed(42)
np.random.seed(42)

# Calendar month of the last index of each dataset's timeline.
ANCHOR_MONTH = {
    "baby": "2014-06",
    "home_and_kitchen": "2014-07",
}

HOLDOUT = 2  # the last two month indices are excluded from the training pool


# ============================================
# MONTH ARITHMETIC
# ============================================

def month_str_to_index(month_str):
    year, month = map(int, str(month_str).split('-'))
    return year * 12 + month


def month_index_to_str(idx):
    year, month = divmod(idx, 12)
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def calendar_index_to_date(idx, max_idx, anchor_month):
    """Global timeline index -> 'YYYY-MM', anchored so that index `max_idx` == anchor_month."""
    anchor_num = month_str_to_index(anchor_month)
    return month_index_to_str(anchor_num - (max_idx - idx))


# ============================================
# TIMEPOINT NAME
# ============================================

def get_timepoint_name_from_delta(history_diff):
    if history_diff == -1:
        return "release_month"
    elif history_diff == 0:
        return "single_month_history"
    elif 1 <= history_diff <= 3:
        return "first_month_to_three_months"
    elif 4 <= history_diff <= 6:
        return "four_months_to_six_months"
    elif 7 <= history_diff <= 12:
        return "seven_months_to_first_year"
    elif 13 <= history_diff <= 24:
        return "first_year_to_second_years"
    elif history_diff > 24:
        return "after_second_years"


# ------------------------------------------
# momentum
# ------------------------------------------

def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unavailable"

    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w + 1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def build_trend_string(counts, begin_month, target_idx, max_idx, anchor_month):
    """'YYYY-MM:value, YYYY-MM:value, ...' for every index in [begin_month, target_idx)."""
    history_idxs = list(range(begin_month, target_idx))
    history_vals = [counts[i] for i in history_idxs]
    if not history_idxs:
        return "No historical trend", history_vals
    trend = ", ".join(
        f"{calendar_index_to_date(i, max_idx, anchor_month)}:{v}"
        for i, v in zip(history_idxs, history_vals)
    )
    return trend, history_vals


# ============================================
# SELECT POPULARITY DATA
# ============================================

def select_popularity_data(counts, begin_month, max_idx, holdout=HOLDOUT):
    last_train_idx = max_idx - holdout

    months_with_details = []
    for target_idx in range(begin_month, last_train_idx + 1):
        delta = target_idx - begin_month
        history_diff = delta - 1
        timepoint_name = get_timepoint_name_from_delta(history_diff)
        months_with_details.append((target_idx, counts[target_idx], delta, timepoint_name))

    release_month_data = None
    other_months = []
    for data in months_with_details:
        if data[2] == 0:
            release_month_data = data
        elif data[2] > 0:
            other_months.append(data)

    selected_data = []
    seen = set()

    def eligible(bucket_name):
        return [m for m in other_months if m[3] == bucket_name and m[0] not in seen]

    if release_month_data:
        selected_data.append(release_month_data[:2] + (release_month_data[3],))
        seen.add(release_month_data[0])

    for m in eligible("single_month_history"):
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    pool = eligible("first_month_to_three_months") or eligible("single_month_history")
    if pool:
        m = random.choice(pool)
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    pool = eligible("four_months_to_six_months") or eligible("first_month_to_three_months")
    if pool:
        m = random.choice(pool)
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    pool = eligible("seven_months_to_first_year") or eligible("four_months_to_six_months")
    if pool:
        m = random.choice(pool)
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    pool = eligible("first_year_to_second_years") or eligible("seven_months_to_first_year")
    if pool:
        m = random.choice(pool)
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    pool = eligible("after_second_years") or eligible("first_year_to_second_years")
    if pool:
        m = random.choice(pool)
        selected_data.append((m[0], m[1], m[3]))
        seen.add(m[0])

    return selected_data


# ============================================
# PER-ROW GENERATION
# ============================================

def build_training_row(item_id, description, counts, begin_month, target_idx, value,
                        timepoint_name, max_idx, anchor_month):
    historical_trend, history_vals = build_trend_string(
        counts, begin_month, target_idx, max_idx, anchor_month
    )
    latest_pop_value = counts[target_idx - 1] if history_vals else np.nan

    return {
        'item_id': item_id,
        'item_description': description,
        'release_month': calendar_index_to_date(begin_month, max_idx, anchor_month),
        'popularity_month': calendar_index_to_date(target_idx, max_idx, anchor_month),
        'popularity_value': value,
        'timepoint_name': timepoint_name,
        'historical_trend': historical_trend,
        'latest_pop_value': latest_pop_value,
        'latest_momentum': calculate_normalized_momentum(history_vals),
    }


def process_dataset(inter_path, meta_path, domain, output_path, label, anchor_month):
    print("=" * 60)
    print(f"{label.upper()}")
    print("=" * 60)

    inter_df = pd.read_csv(inter_path)
    inter_df['inter_month_count'] = inter_df['inter_month_count'].apply(ast.literal_eval)

    seq_lens = inter_df['inter_month_count'].apply(len)
    assert seq_lens.nunique() == 1, (
        f"{label}: inter_month_count sequences have inconsistent lengths ({seq_lens.unique()})"
    )
    max_idx = int(seq_lens.iloc[0]) - 1
    print(f"Derived max_idx={max_idx} from data (sequence length {max_idx + 1})")

    descriptions, n_meta_total, missingness_report = build_descriptions_from_meta(meta_path, domain)
    print(f"Meta file: {n_meta_total} items total")
    for field, n_missing in missingness_report.items():
        print(f"  {field} missing: {n_missing} / {n_meta_total} ({n_missing / n_meta_total:.1%})")

    n_with_desc = inter_df['item_id'].isin(descriptions).sum()
    print(f"Inter-catalog items: {len(inter_df)} | with a metadata row at all: "
          f"{n_with_desc} ({n_with_desc / len(inter_df):.1%})")

    rows = []
    excluded_no_meta = []
    excluded_no_selection = []
    for _, row in inter_df.iterrows():
        item_id = row['item_id']
        counts = row['inter_month_count']
        begin_month = row['begin_month']
        description = descriptions.get(item_id)

        if description is None:
            excluded_no_meta.append(item_id)
            continue

        selections = select_popularity_data(counts, begin_month, max_idx, holdout=HOLDOUT)
        if not selections:
            excluded_no_selection.append(item_id)
            continue

        for target_idx, value, timepoint_name in selections:
            rows.append(build_training_row(
                item_id, description, counts, begin_month, target_idx, value,
                timepoint_name, max_idx, anchor_month,
            ))

    out_df = pd.DataFrame(rows)
    n_covered = out_df['item_id'].nunique() if len(out_df) else 0
    print(f"HOLDOUT={HOLDOUT} (last training index = max_idx - {HOLDOUT} = {max_idx - HOLDOUT})")
    print(f"Items without a metadata row: {len(excluded_no_meta)} / {len(inter_df)}")
    print(f"Items with no eligible training month: {len(excluded_no_selection)} / {len(inter_df)}")
    print(f"Generated {len(out_df)} training rows from {n_covered} items "
          f"({out_df.groupby('item_id').size().mean():.2f} rows/item on average)"
          if len(out_df) else "Generated 0 training rows")
    if len(out_df):
        print(out_df['timepoint_name'].value_counts())

    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")
    return out_df


# ============================================
# JSON-FORMAT TREND (test set)
# ============================================

def build_trend_json(counts, begin_month, target_idx, max_idx, anchor_month):
    """List of {"timestamp", "popularity_score"} for every index in [begin_month, target_idx)."""
    history_idxs = list(range(begin_month, target_idx))
    pairs = [
        {"timestamp": calendar_index_to_date(i, max_idx, anchor_month), "popularity_score": counts[i]}
        for i in history_idxs
    ]
    return pairs


# ============================================
# TEST SET
# ============================================

def build_test_set(inter_path, meta_path, domain, output_path, label, anchor_month, truncate_last_n=0):
    """One row per item with target = the last index (after dropping `truncate_last_n`
    trailing months). Columns: item_id, product_description, predict_timestamp,
    historical_popularity_trends (JSON list), ground_truth_popularity_score.

    Items with no activity in the target month are excluded. Debut items (no activity
    before the held-out window) are kept and logged separately.
    """
    print("=" * 60)
    print(f"{label.upper()} -- TEST SET")
    print("=" * 60)

    inter_df = pd.read_csv(inter_path)
    inter_df['inter_month_count'] = inter_df['inter_month_count'].apply(ast.literal_eval)

    if truncate_last_n:
        inter_df['inter_month_count'] = inter_df['inter_month_count'].apply(
            lambda lst: lst[:len(lst) - truncate_last_n]
        )

    seq_lens = inter_df['inter_month_count'].apply(len)

    assert seq_lens.nunique() == 1, (
        f"{label} test set: inter_month_count sequences have inconsistent lengths ({seq_lens.unique()})"
    )
    max_idx = int(seq_lens.iloc[0]) - 1
    print(f"Derived max_idx={max_idx} (test target index); anchor_month={anchor_month}")

    descriptions, n_meta_total, missingness_report = build_descriptions_from_meta(meta_path, domain)
    print(f"Meta file: {n_meta_total} items total")

    meta_df = pd.read_csv(meta_path, usecols=["item_id", "item"])
    id_to_asin = meta_df.set_index("item_id")["item"].to_dict()

    rows = []
    excluded_no_meta = []
    excluded_zero_test_month = []
    debut_items_kept = []
    for _, row in inter_df.iterrows():
        item_id = row['item_id']
        counts = row['inter_month_count']
        begin_month = row['begin_month']
        description = descriptions.get(item_id)

        if description is None:
            excluded_no_meta.append(item_id)
            continue

        target_idx = max_idx
        value = counts[target_idx]

        # exclude items with no activity in the target month
        if value <= 0:
            excluded_zero_test_month.append({
                "item_id": item_id,
                "item": id_to_asin.get(item_id),
                "begin_month": begin_month,
                "value_at_test_month": value,
                "last_5_counts": counts[max(begin_month, max_idx - 4): max_idx + 1],
            })
            continue

        # debut items (no prior activity) are kept but logged
        prior_counts = counts[begin_month: max_idx - HOLDOUT + 1]
        if not prior_counts or max(prior_counts) <= 0:
            debut_items_kept.append({
                "item_id": item_id,
                "item": id_to_asin.get(item_id),
                "begin_month": begin_month,
                "value_at_test_month": value,
                "prior_window_counts": prior_counts,
            })

        rows.append({
            "item_id": item_id,
            "product_description": description,
            "predict_timestamp": calendar_index_to_date(target_idx, max_idx, anchor_month),
            "historical_popularity_trends": json.dumps(
                build_trend_json(counts, begin_month, target_idx, max_idx, anchor_month)
            ),
            "ground_truth_popularity_score": value,
        })

    out_df = pd.DataFrame(rows)
    print(f"Items without a metadata row: {len(excluded_no_meta)} / {len(inter_df)}")
    print(f"Items with zero activity in the test month (excluded): "
          f"{len(excluded_zero_test_month)} / {len(inter_df)}")
    print(f"Debut items (kept): {len(debut_items_kept)} / {len(inter_df)}")
    print(f"Generated {len(out_df)} test rows")
    
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    base = output_path.rsplit(".", 1)[0]
    pd.DataFrame(excluded_zero_test_month).to_csv(f"{base}_excluded_zero_test_month.csv", index=False)
    pd.DataFrame(debut_items_kept).to_csv(f"{base}_debut_items_kept.csv", index=False)
    print("Saved exclusion/debut diagnostics alongside the main output.\n")

    return out_df

def main():
    # Baby is generated the same way:
    # process_dataset(
    #     inter_path='dataset/Amazon14/inter_data/Baby_inter_train.csv',
    #     meta_path='dataset/Amazon14/metadata/Baby_meta.csv',
    #     domain='baby',
    #     output_path='dataset/Amazon14/synthetic_data/baby_synthetic_training_data.csv',
    #     label='baby',
    #     anchor_month=ANCHOR_MONTH['baby'],
    # )

    process_dataset(
        inter_path='dataset/Amazon14/inter_data/Home_and_Kitchen_inter_train.csv',
        meta_path='dataset/Amazon14/metadata/Home_and_Kitchen_meta.csv',
        domain='home and kitchen',
        output_path='dataset/Amazon14/synthetic_data/260921_home_and_kitchen_synthetic_training_data.csv',
        label='home_and_kitchen',
        anchor_month=ANCHOR_MONTH['home_and_kitchen'],
    )

    # test set: the last month is dropped (truncate_last_n=1), so anchor_month is that of the new last index
    build_test_set(
        inter_path='dataset/Amazon14/inter_data/Home_and_Kitchen_inter_train.csv',
        meta_path='dataset/Amazon14/metadata/Home_and_Kitchen_meta.csv',
        domain='home and kitchen',
        output_path='dataset/Amazon14/synthetic_data/260921_home_and_kitchen_validation_data.csv',
        label='home_and_kitchen_test',
        anchor_month='2014-06',
        truncate_last_n=1,
    )


if __name__ == "__main__":
    main()