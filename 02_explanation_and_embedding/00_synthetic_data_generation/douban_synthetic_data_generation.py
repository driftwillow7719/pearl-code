"""
Synthetic training rows for Douban movies: for each movie, one target month per history-depth
bucket (release month, single-month history, 1-3, 4-6, 7-12, 13-24, >24 months) is selected,
and each row gets the historical trend, trend array, latest popularity value, momentum and item age.

Usage: python douban_synthetic_data_generation.py --input <train_data.xlsx> --output <out.csv>
"""
import pandas as pd
import numpy as np
from datetime import datetime
import random
import re
# fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ============================================
# HELPERS
# ============================================
def calculate_item_age(row):
    """Age in months of the latest known data point (one month before popularity_month)."""
    try:
        release_dt = datetime.strptime(str(row['release_date']), "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

    try:
        pop_year, pop_month_num = map(int, str(row['popularity_month']).split('-'))
    except (ValueError, TypeError):
        return None

    latest_known_year, latest_known_month = pop_year, pop_month_num - 1
    if latest_known_month == 0:
        latest_known_month = 12
        latest_known_year -= 1

    return (latest_known_year - release_dt.year) * 12 + (latest_known_month - release_dt.month)

# ------------------------------------------
# create trend arrays
# ------------------------------------------
def extract_popularity_counts(historical_trend_str):
    """Extract the popularity counts from a historical trend string."""
    if pd.isna(historical_trend_str) or historical_trend_str == '':
        return np.array([])
    
    # Extract numbers after colon
    values = re.findall(r':\s*(\d+)', historical_trend_str)
    return np.array([int(v) for v in values])

def create_trend_arrays(df,historical_trend_str):
    """Add trend_array and drop the YYYY-MM columns."""
    df['trend_array'] = df[historical_trend_str].apply(extract_popularity_counts)
    date_pattern = re.compile(r'^\d{4}-\d{2}$')
    date_columns = [col for col in df.columns if date_pattern.match(str(col))]
    df = df.drop(columns=date_columns)
    print(f"✅ Created 'trend_array' column and removed date columns.")
    return df
# ------------------------------------------
# latest pop value
# ------------------------------------------
def extract_latest_popularity(historical_trend_str):
    """Latest popularity value of a trend string, e.g. "2014-08: 11, 2014-09: 2" -> 2."""
    if pd.isna(historical_trend_str) or historical_trend_str == '' or historical_trend_str == 'No historical trend':
        return np.nan
    
    pattern = r':\s*(-?\d+)$'
    match = re.search(pattern, str(historical_trend_str).strip())
    
    if match:
        return int(match.group(1))
    
    return np.nan
# ------------------------------------------
# momentum
# ------------------------------------------
def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unavailable"

    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w+1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0

def get_timepoint_name_from_delta(history_diff):
    """Bucket by the age (months since release) of the latest historical point, which is
    one month before the target month: -1 = release month (no history), 0 = one point,
    then 1-3, 4-6, 7-12, 13-24 and >24 months."""
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


def get_trend_from_release_to_target(row, target_month, popularity_start_col=12):
    """Popularity trend of a movie from its release month up to (excluding) the target month."""
    try:
        raw_date = str(row['release_date'])
        if '/' in raw_date:
            release_date = datetime.strptime(raw_date, "%d/%m/%Y %H.%M.%S")
        else:
            release_date = datetime.strptime(raw_date.split()[0], "%Y-%m-%d")

        release_month = release_date.strftime("%Y-%m")
    except Exception:
        return f"{target_month}: {row.get(target_month, -1)}", True

    if release_month >= target_month:
        return "No historical trend", True

    trend_months = []
    is_valid = True

    for col_name in row.index[popularity_start_col:]:
        if '-' not in col_name or len(col_name) != 7:
            continue

        try:
            value = row[col_name]
            if pd.isna(value) or value == "":
                continue

            value = float(value)

            if release_month <= col_name < target_month:
                trend_months.append((col_name, value))

        except (ValueError, TypeError):
            continue

    if not trend_months:
        return "No historical trend", is_valid

    trend_months.sort(key=lambda x: x[0])
    trend_parts = [f"{month}:{int(value) if value.is_integer() else round(value, 2)}"
                   for month, value in trend_months]

    return ", ".join(trend_parts), is_valid



# ============================================
# SELECT POPULARITY DATES AND VALUES
# ============================================

def select_popularity_data(row, popularity_start_col=12):
    """Select target months, one per history-depth bucket (see get_timepoint_name_from_delta).

    Returns a list of (popularity_date, popularity_value, timepoint_name, historical_trend, is_valid).
    """

    try:
        release_datetime = datetime.strptime(str(row['release_date']), "%Y-%m-%d %H:%M:%S")
        release_date = release_datetime.date()
        release_month = release_date.strftime("%Y-%m")
        release_year, release_month_num = map(int, release_month.split('-'))
    except:
        release_month = None
        release_year, release_month_num = None, None
    popularity_months = []
    for i in range(popularity_start_col, len(row)):
        col_name = row.index[i]
        if pd.notna(row.iloc[i]) and row.iloc[i] != "":
            try:
                if '-' in col_name and len(col_name) == 7:
                    year, month = col_name.split('-')
                    if year.isdigit() and month.isdigit():
                        popularity_months.append((col_name, float(row.iloc[i])))
            except (ValueError, TypeError):
                continue

    months_with_details = []
    for month, level in popularity_months:
        try:
            year, month_num = map(int, month.split('-'))
            if release_year is not None and release_month_num is not None:
                delta = (year - release_year) * 12 + (month_num - release_month_num)
                history_diff = delta - 1
                timepoint_name = get_timepoint_name_from_delta(history_diff)
                months_with_details.append((month, level, delta, timepoint_name))
        except:
            continue

    release_month_data = None
    other_months = []

    for data in months_with_details:
        if data[0] == release_month:
            release_month_data = data
        elif data[2] > 0:  # Only post-release months
            other_months.append(data)

    selected_data = []
    seen_months = set()

    # release month (no history) is always kept
    if release_month_data:
        trend, is_valid = get_trend_from_release_to_target(row, release_month_data[0], popularity_start_col)
        selected_data.append((release_month_data[0], release_month_data[1], release_month_data[3], trend, is_valid))
        seen_months.add(release_month_data[0])

    # single_month_history: keep all eligible rows
    eligible_single = [m for m in other_months if m[3] == "single_month_history" and m[0] not in seen_months]
    for selected in eligible_single:
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    # first_month_to_three_months
    eligible_1_3 = [m for m in other_months if m[3] == "first_month_to_three_months" and m[0] not in seen_months]
    if not eligible_1_3:
        eligible_1_3 = [m for m in other_months if m[3] == "single_month_history" and m[0] not in seen_months]
    if eligible_1_3:
        selected = random.choice(eligible_1_3)
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    # four_months_to_six_months
    eligible_4_6 = [m for m in other_months if m[3] == "four_months_to_six_months" and m[0] not in seen_months]
    if not eligible_4_6:
        eligible_4_6 = [m for m in other_months if m[3] == "first_month_to_three_months" and m[0] not in seen_months]

    if eligible_4_6:
        selected = random.choice(eligible_4_6)
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    # seven_months_to_first_year
    eligible_7_12 = [m for m in other_months if m[3] == "seven_months_to_first_year" and m[0] not in seen_months]
    if not eligible_7_12:
        eligible_7_12 = [m for m in other_months if m[3] == "four_months_to_six_months" and m[0] not in seen_months]

    if eligible_7_12:
        selected = random.choice(eligible_7_12)
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    # first_year_to_second_years
    eligible_13_24 = [m for m in other_months if m[3] == "first_year_to_second_years" and m[0] not in seen_months]
    if not eligible_13_24:
        eligible_13_24 = [m for m in other_months if m[3] == "seven_months_to_first_year" and m[0] not in seen_months]

    if eligible_13_24:
        selected = random.choice(eligible_13_24)
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    # after_second_years
    eligible_24_plus = [m for m in other_months if m[3] == "after_second_years" and m[0] not in seen_months]
    if not eligible_24_plus:
        eligible_24_plus = [m for m in other_months if m[3] == "first_year_to_second_years" and m[0] not in seen_months]

    if eligible_24_plus:
        selected = random.choice(eligible_24_plus)
        trend, is_valid = get_trend_from_release_to_target(row, selected[0], popularity_start_col)
        selected_data.append((selected[0], selected[1], selected[3], trend, is_valid))
        seen_months.add(selected[0])

    return selected_data

# ============================================
# HISTORICAL TREND
# ============================================

def add_historical_trend_column(df, popularity_start_col=12):
    """Wide movie table -> long format, one row per selected target month."""
    all_rows = []

    for idx, row in df.iterrows():
        selected_timepoints = select_popularity_data(row, popularity_start_col)

        for target_month, target_value, timepoint_name, historical_trend, is_valid in selected_timepoints:
            row_data = row.to_dict()

            row_data['popularity_month'] = target_month
            row_data['popularity_value'] = int(target_value) if target_value.is_integer() else round(target_value, 2)
            row_data['timepoint_name'] = timepoint_name
            row_data['historical_trend'] = historical_trend
            row_data['is_valid'] = is_valid

            all_rows.append(row_data)

    long_df = pd.DataFrame(all_rows)

    if long_df.empty:
        return long_df

    target_cols = ['popularity_month', 'popularity_value', 'timepoint_name', 'historical_trend', 'is_valid']
    other_cols = [col for col in long_df.columns if col not in target_cols]

    return long_df[other_cols + target_cols]

# ============================================
# CLEANING AND MAIN
# ============================================

def final_data_cleaning(df):
    """Drop the YYYY-MM columns, 'imdb' and 'is_valid', and rows with missing values."""
    pattern = r'^\d{4}-\d{2}$'
    columns_to_drop = [col for col in df.columns if re.match(pattern, str(col))]

    df_cleaned = df.drop(columns=columns_to_drop)

    print(f"Dropped {len(columns_to_drop)} columns with YYYY-MM format")
    if len(columns_to_drop) > 0:
        print(f"Columns dropped: {columns_to_drop[:10]}", end="")
        if len(columns_to_drop) > 10:
            print(f"... and {len(columns_to_drop) - 10} more")
        else:
            print()

    df_cleaned = df.drop(columns=columns_to_drop + ['imdb', 'is_valid'])
    df_cleaned['release_date'] = df_cleaned['release_date'].dt.strftime('%Y-%m-%d')
    df_cleaned_final = df_cleaned.dropna().reset_index(drop=True)
    print(f"Removed {len(df_cleaned) - len(df_cleaned_final)} rows with missing values")
    return df_cleaned_final

def main(input_path, output_path):
    print("MOVIE POPULARITY DATA PROCESSING")

    df = pd.read_excel(input_path)
    df = df.rename(columns={'movieid': 'movie_id'})
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # movies released before this date are old enough to reach the after_second_years bucket
    release_dates = pd.to_datetime(df['release_date'])
    cutoff = pd.Timestamp('2017-09-10')
    eligible = (release_dates < cutoff).sum()
    total = len(release_dates)
    print(f"Movies old enough to ever reach after_second_years: {eligible} / {total} ({eligible/total:.1%})")

    popularity_start_col = 12  # index where the popularity columns start
    df_with_trend = add_historical_trend_column(df, popularity_start_col)
    print(df_with_trend[['movie_id', 'movie_name', 'historical_trend', 'is_valid']].head())

    cleaned_long_df = final_data_cleaning(df_with_trend)
    cleaned_long_df['item_age'] = cleaned_long_df.apply(calculate_item_age, axis=1)
    cleaned_long_df = create_trend_arrays(cleaned_long_df, 'historical_trend')
    cleaned_long_df['latest_pop_value'] = cleaned_long_df['historical_trend'].apply(extract_latest_popularity).astype("Int64")
    cleaned_long_df['latest_momentum'] = cleaned_long_df['trend_array'].apply(calculate_normalized_momentum)

    cleaned_long_df.to_csv(output_path, index=False)
    print(cleaned_long_df.info())
    print(f"File saved: {output_path}")
    return df_with_trend


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the Douban synthetic training data.")
    parser.add_argument("--input", required=True, help="processed complete train data (.xlsx)")
    parser.add_argument("--output", required=True, help="output CSV")
    args = parser.parse_args()
    main(args.input, args.output)
