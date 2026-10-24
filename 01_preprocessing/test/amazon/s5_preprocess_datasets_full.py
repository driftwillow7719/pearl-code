"""
Feature engineering for the Amazon baby/kitchen.

Each TEST row already carries the item's full pre-test history and the true
test-month outcome:

    item_id, product_description, predict_timestamp (always "2014-06" here),
    historical_popularity_trends (JSON list of {timestamp, popularity_score},
    strictly before predict_timestamp), ground_truth_popularity_score
"""

import json
import re

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r'^\d{4}-\d{2}$')

# Fixed validation predict_timestamp -- one month before the TEST set's
# "2014-06", per 260918_amazon_validation_data_generation.py's target
# selection (MAX_IDX - 1, reserving MAX_IDX/2014-06 for the real test set).
VAL_PREDICT_TIMESTAMP = "2014-05"

# ============================================
# MONTH ARITHMETIC
# ============================================

def month_str_to_index(month_str):
    year, month = map(int, str(month_str).split('-'))
    return year * 12 + month


def month_diff(month_from, month_to):
    return month_str_to_index(month_to) - month_str_to_index(month_from)


# ============================================
# PARSING historical_popularity_trends (TEST set: JSON)
# ============================================

def parse_trend_json(trend_str):
    """historical_popularity_trends JSON string -> chronologically sorted
    list of (month_str, score) tuples."""
    if pd.isna(trend_str) or str(trend_str).strip() in ('', '[]'):
        return []
    try:
        records = json.loads(trend_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    pairs = []
    for rec in records:
        try:
            ts = str(rec['timestamp'])
            score = float(rec['popularity_score'])
        except (KeyError, TypeError, ValueError):
            continue
        if MONTH_RE.match(ts):
            pairs.append((ts, score))
    pairs.sort(key=lambda x: x[0])
    return pairs


def format_trend_string(pairs):
    """[(month, value), ...] -> 'YYYY-MM:value, YYYY-MM:value, ...'."""
    if not pairs:
        return "No historical trend"
    return ", ".join(f"{month}:{int(v) if float(v).is_integer() else v}" for month, v in pairs)


# ============================================
# PARSING historical_trend (VALIDATION set: plain 'YYYY-MM:value' string --
# this IS the format 260918_amazon_validation_data_generation.py already
# writes, so this is just the inverse of format_trend_string above).
# ============================================

def parse_trend_string(trend_str):
    """'YYYY-MM:value, YYYY-MM:value, ...' -> chronologically sorted list of
    (month_str, value) tuples. 'No historical trend' / blank / NaN -> []."""
    if pd.isna(trend_str) or str(trend_str).strip() in ('', 'No historical trend'):
        return []

    pairs = []
    for part in str(trend_str).split(','):
        part = part.strip()
        if not part or ':' not in part:
            continue
        month, val = part.split(':', 1)
        month = month.strip()
        if not MONTH_RE.match(month):
            continue
        try:
            pairs.append((month, float(val.strip())))
        except ValueError:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


# ============================================
# LIFECYCLE FEATURES (ported from s5_save_eight_groups.py, unchanged)
# ============================================

def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unavailable"

    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w + 1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def trend_summary_stats(trend_array):
    """trend_min / trend_max / trend_volatility / time_since_peak, with the
    same NaN-for-unmeasurable rules as s5_save_eight_groups.py
    (trend_overall_change is computed too, for parity, though it wasn't in
    the requested column list)."""
    arr = np.asarray(trend_array, dtype=float) if isinstance(trend_array, (list, np.ndarray)) else None

    if arr is None or len(arr) == 0:
        return {
            'trend_min': np.nan,
            'trend_max': np.nan,
            'trend_volatility': np.nan,
            'trend_overall_change': np.nan,
            'time_since_peak': np.nan,
        }

    trend_min = float(arr.min())
    trend_max = float(arr.max())
    trend_volatility = float(arr.std()) if len(arr) > 1 else np.nan
    trend_overall_change = float(arr[-1] - arr[0]) if len(arr) > 1 else np.nan

    # well-defined even for a single point: that point is trivially both
    # the peak and the most recent value
    peak_idx = int(arr.argmax())
    time_since_peak = (len(arr) - 1) - peak_idx

    return {
        'trend_min': trend_min,
        'trend_max': trend_max,
        'trend_volatility': trend_volatility,
        'trend_overall_change': trend_overall_change,
        'time_since_peak': time_since_peak,
    }


# ============================================
# PER-ROW PROCESSING -- TEST set (parses JSON, (re)builds historical_trend
# string from scratch)
# ============================================

def process_row(pairs, predict_timestamp):
    trend_array = np.array([v for _, v in pairs], dtype=int)

    if not pairs:
        # true debut at test time: no history at all
        return {
            'release_month': None,
            'item_age': -1,
            'historical_trend': "No historical trend",
            'trend_array': trend_array,
            'latest_pop_value': np.nan,
            'latest_momentum': "unavailable",
            **trend_summary_stats(trend_array),
        }

    release_month = pairs[0][0]
    item_age = month_diff(release_month, predict_timestamp) - 1
    latest_pop_value = pairs[-1][1]

    return {
        'release_month': release_month,
        'item_age': item_age,
        'historical_trend': format_trend_string(pairs),
        'trend_array': trend_array,
        'latest_pop_value': latest_pop_value,
        'latest_momentum': calculate_normalized_momentum(trend_array),
        **trend_summary_stats(trend_array),
    }


def process_test_set(input_path, output_path, label):
    print("=" * 60)
    print(f"{label.upper()} TEST SET")
    print("=" * 60)

    # Baby's original test files were .xlsx; home_and_kitchen's
    # build_test_set() output is .csv
    # directly -- pick the reader by extension so this works for either.
    df = pd.read_excel(input_path) if str(input_path).endswith(('.xlsx', '.xls')) else pd.read_csv(input_path)
    print(f"Loaded: {len(df)} rows")

    feature_rows = []
    for _, row in df.iterrows():
        pairs = parse_trend_json(row['historical_popularity_trends'])
        feature_rows.append(process_row(pairs, row['predict_timestamp']))

    features_df = pd.DataFrame(feature_rows)
    out_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    n_debut = (out_df['item_age'] == -1).sum()
    n_single = (out_df['item_age'] == 0).sum()
    n_mom_unavailable = (out_df['latest_momentum'] == 'unavailable').sum()
    print(f"Debut (zero history) rows: {n_debut}")
    print(f"Single-point-history rows: {n_single}")
    print(f"Momentum unavailable (fewer than 2 real points): {n_mom_unavailable}")
    print(f"item_age range: {out_df['item_age'].min()} to {out_df['item_age'].max()}")

    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")
    return out_df


# ============================================
# PER-ROW PROCESSING -- VALIDATION set. Only what's actually available
# there: historical_trend is ALREADY a formatted string (not JSON), and
# latest_pop_value/latest_momentum are ALREADY computed (kept as-is, not
# recomputed) -- this function only adds what the validation candidate
# list doesn't already carry: release_month, item_age, and the
# trend_min/max/volatility/overall_change/time_since_peak summary stats,
# all derived by parsing that same historical_trend string back into a
# trend_array. predict_timestamp is fixed (VAL_PREDICT_TIMESTAMP), since
# the validation candidate list doesn't carry a per-row timestamp column.
# ============================================

def process_val_row(pairs, predict_timestamp, historical_trend, latest_pop_value, latest_momentum):
    trend_array = np.array([v for _, v in pairs], dtype=int)

    if not pairs:
        return {
            'release_month': None,
            'item_age': -1,
            'historical_trend': historical_trend,
            'trend_array': trend_array,
            'latest_pop_value': latest_pop_value,
            'latest_momentum': latest_momentum,
            **trend_summary_stats(trend_array),
        }

    release_month = pairs[0][0]
    item_age = month_diff(release_month, predict_timestamp) - 1

    return {
        'release_month': release_month,
        'item_age': item_age,
        'historical_trend': historical_trend,
        'trend_array': trend_array,
        'latest_pop_value': latest_pop_value,
        'latest_momentum': latest_momentum,
        **trend_summary_stats(trend_array),
    }


def process_val_set(input_path, output_path, label, predict_timestamp=VAL_PREDICT_TIMESTAMP):
    print("=" * 60)
    print(f"{label.upper()} VALIDATION SET")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} rows")

    feature_rows = []
    for _, row in df.iterrows():
        pairs = parse_trend_string(row['historical_trend'])
        feature_rows.append(process_val_row(
            pairs, predict_timestamp,
            historical_trend=row['historical_trend'],
            latest_pop_value=row['latest_pop_value'],
            latest_momentum=row['latest_momentum'],
        ))

    features_df = pd.DataFrame(feature_rows)
    base_df = df.drop(columns=['historical_trend','release_month','latest_pop_value', 'latest_momentum']).reset_index(drop=True)
    out_df = pd.concat([base_df, features_df], axis=1)
    out_df = out_df.rename(columns={'popularity_value': 'ground_truth_popularity_score'})
    out_df['predict_timestamp'] = predict_timestamp

    n_debut = (out_df['item_age'] == -1).sum()
    n_single = (out_df['item_age'] == 0).sum()
    n_mom_unavailable = (out_df['latest_momentum'] == 'unavailable').sum()
    print(f"Debut (zero history) rows: {n_debut}")
    print(f"Single-point-history rows: {n_single}")
    print(f"Momentum unavailable (fewer than 2 real points): {n_mom_unavailable}")
    print(f"item_age range: {out_df['item_age'].min()} to {out_df['item_age'].max()}")

    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")
    return out_df


def main():
    # process_test_set(
    #     input_path='datasets/amazon_baby_test/last_month_amazon_baby_dataset.xlsx',
    #     output_path='datasets/amazon_baby_test/baby_test_features_2.csv',
    #     label='baby',
    # )
    
    # Amazon Kitchen's test set is named validation set because PARE's validation set window is on JUNE 2016, the same as TRAIL's test set
    process_test_set(
        input_path='PARE/dataset/Amazon14/synthetic_data/kitchen_home/260921_home_and_kitchen_validation_data.csv',
        output_path='PARE/dataset/Amazon14/synthetic_data/kitchen_home/260921_home_and_kitchen_val_features.csv',
        label='home_and_kitchen',
    )
    # process_val_set(
    #     input_path='260918_baby_validation_candidates.csv',
    #     output_path='260918_baby_validation_features.csv',
    #     label='baby',
    # )


if __name__ == "__main__":
    main()
