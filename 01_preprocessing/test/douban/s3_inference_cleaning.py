# for cleaning the inference data to prepare clustering model applications
import pandas as pd
import numpy as np
import re
from clean_data_utils import clean_text,clean_column_text, clean_column_duplicates,clean_cast
import argparse
# ---------------------
# data loading
# ---------------------
def data_loading(filepath):
    df = pd.read_excel(filepath)
    df['release_date'] = df['release_date'].dt.strftime('%Y-%m-%d')
    return df
# ---------------------
# basic cleaning and remove data out of scope
# ---------------------
def data_cleaning(df,indicator):
    """Clean multiple text columns at once"""
    text_columns = ['producers', 'top5_actors', 'director', 'writers', 'music']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            df = clean_column_duplicates(df, col, ',')
            if col == 'top5_actors':
                df[col] = df[col].apply(lambda x: clean_cast(x, max_actors=5))
            else:
                df[col] = df[col].apply(clean_text)
    if 'description' in df.columns:
        df = clean_column_text(df,'description')
    if indicator == 'validation_set':
        date_filter = df['release_date'] < '2019-10-01'
        filtered_df = df[date_filter].copy()
        non_df = df[~date_filter].copy()
        print(f"Filtered out {len(df) - len(filtered_df)} movies released after Oct 2019 for validation set.")
        print(f"Sample non-eligible movies:")
        print(non_df.head())
        return filtered_df
    elif indicator == 'test_set':    
        date_filter = df['release_date'] < '2019-12-01'
        filtered_df = df[date_filter].copy()
        print(f"Filtered out {len(df) - len(filtered_df)} movies released after Nov 2019 for test set.")
        return filtered_df


def create_popularity_month_value(df,indicator):
    if indicator == 'validation_set':
        df['popularity_month'] = '2019-09'
        df['popularity_value'] = df['2019-09']
    elif indicator == 'test_set':
        df['popularity_month'] = '2019-10'
        df['popularity_value'] = df['2019-10']
    else:
        df['popularity_month'] = 'unknown'
        df['popularity_value'] = -1
    return df


# ------------------------------------------
# get item age
# ------------------------------------------
def calculate_item_age(row):
    # Debut case: no popularity data has ever been observed for this item
    if pd.isna(row.get('latest_pop_value')) and row.get('historical_trend', '') == '':
        return -1
    try:
        release_dt = pd.to_datetime(str(row['release_date']))
        release_year = release_dt.year
        release_month_num = release_dt.month
    except (ValueError, TypeError):
        return None

    pop_year, pop_month_num = map(int, str(row['popularity_month']).split('-'))
    delta = (pop_year - release_year) * 12 + (pop_month_num - release_month_num)
    return delta

# ------------------------------------------
# get latest pop value
# ------------------------------------------

def extract_latest_popularity(historical_trend_str):
    """
    Take latest pop score
    Example: "2014-08: 11, 2014-09: 2" -> 2
    """
    # Handle if it's empty or No historical trend
    if pd.isna(historical_trend_str) or historical_trend_str == '' or historical_trend_str == 'No historical trend':
        return np.nan
    
    # Pattern: find optional negative sign + digits, after ':' (with optional space), at the end of the string
    pattern = r':\s*(-?\d+)$'
    match = re.search(pattern, str(historical_trend_str).strip())
    
    if match:
        return int(match.group(1))
    
    return np.nan


def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unavailable"

    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    # average log1p difference over the trailing window, not just the last step
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w+1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0

# ------------------------------------------
# get timepoint name
# ------------------------------------------
def get_months_diff_from_yyyy_mm(date1, date2):
    """
    Hitung selisih bulan antara dua tanggal format YYYY-MM
    date2 - date1
    
    Contoh:
    date1 = '2024-01', date2 = '2024-06' -> return 5
    """
    year1, month1 = map(int, date1.split('-'))
    year2, month2 = map(int, date2.split('-'))
    
    return (year2 - year1) * 12 + (month2 - month1)

def get_timepoint_name_from_delta(months_diff):
    """Determines the timepoint name based on months difference."""
    if months_diff == 0:
        return "release_month"
    elif 1 <= months_diff <= 3:
        return "first_month_to_three_months"
    elif 4 <= months_diff <= 6:
        return "four_months_to_six_months"
    elif 7 <= months_diff <= 12:
        return "seven_months_to_first_year"
    elif 13 <= months_diff <= 24:
        return "first_year_to_second_years"
    elif months_diff > 24:
        return "after_second_years"
# ------------------------------------------
# get popularity trend
# ------------------------------------------

def get_trend_from_release_to_popularity(row, popularity_start_col=12):
    """
    Get historical trend from release month to popularity month.
    
    Returns:
        historical_trend (str): "YYYY-MM: value, YYYY-MM: value, ..."
        is_valid (bool): True if no -1 values
        timepoint_name (str): category based on months difference
    """
    
    # 1. Get release_month from release_date
    try:
        release_date = pd.to_datetime(row['release_date'])
        release_month = release_date.strftime("%Y-%m")
    except:
        return "", False, "unknown"
    
    # 2. Get target_month
    target_month = row.get('popularity_month', 'unknown')
    if target_month == 'unknown' or pd.isna(target_month):
        return "", False, "unknown"
    
    # 3. Hitung months_diff
    try:
        r_year, r_month = map(int, release_month.split('-'))
        t_year, t_month = map(int, target_month.split('-'))
        months_diff = (t_year - r_year) * 12 + (t_month - r_month)
    except:
        return "", False, "unknown"
    
    # 4. Get timepoint_name: shows true age of the item
    timepoint_name = get_timepoint_name_from_delta(months_diff)
    
    if months_diff < 0:
        return "", False, timepoint_name
    
    # 5. Find column positions
    date_columns = [col for col in row.index[popularity_start_col:] 
                    if '-' in col and len(col) == 7]
    
    if not date_columns:
        return "", False, timepoint_name
    
    # 6. Find start_pos and end_pos
    # START POS with fallback
    try:
        start_pos = date_columns.index(release_month)
    except ValueError:
        # Fallback: start from 2010-01 or any available month
        if '2010-01' in date_columns:
            start_pos = date_columns.index('2010-01')
        else:
            start_pos = 0
        print(f"⚠️ Release month '{release_month}' not found, using start at {date_columns[start_pos]}")
    
    # END POS with fallback
    try:
        end_pos = date_columns.index(target_month)
    except ValueError:
        # Fallback: use the last available month
        end_pos = len(date_columns) - 1
        print(f"⚠️ Target month '{target_month}' not found, using end at {date_columns[end_pos]}")
    
    # 7. Get trend from available start_pos to end_pos
    trend_months = []
    is_valid = True
    
    for i in range(start_pos, end_pos + 1):
        month = date_columns[i]
        value = row.get(month)
        
        if pd.isna(value) or value == "":
            continue
        
        try:
            value = float(value)
            trend_months.append((month, value))
            if value == -1:
                is_valid = False
        except (ValueError, TypeError):
            continue
    
    # 8. Format output
    if not trend_months:
        return "", False, timepoint_name
    
    trend_parts = [f"{month}: {int(value) if value.is_integer() else round(value, 2)}" 
                    for month, value in trend_months]
    
    return ", ".join(trend_parts), is_valid, timepoint_name

# ------------------------------------------
# get historical trend column
# ------------------------------------------
def add_historical_trend_column(df):
    """
    Adds historical_trend column to dataframe using the two functions above.
    
    Parameters:
    - df: Input dataframe
    - popularity_start_col: Index of first popularity column
    
    Returns:
    - Dataframe with new 'historical_trend' and 'is_valid' columns
    """
    df_copy = df.copy()
    trend_results = df_copy.apply(get_trend_from_release_to_popularity, axis=1)
    df_copy['historical_trend'] = trend_results.apply(lambda x: x[0])
    df_copy['is_valid'] = trend_results.apply(lambda x: x[1])
    df_copy['timepoint_name'] = trend_results.apply(lambda x: x[2])
    return df_copy

# ------------------------------------------
# get array version of historical trend
# ------------------------------------------
def extract_popularity_counts(historical_trend_str):
    """Extract popularity counts from the historical trend string"""
    if pd.isna(historical_trend_str) or historical_trend_str == '':
        return np.array([])
    
    # Extract numbers after colon
    values = re.findall(r':\s*(\d+)', historical_trend_str)
    return np.array([int(v) for v in values])

# ------------------------------------------
# create trend arrays
# ------------------------------------------
def create_trend_arrays(df,historical_trend_str):
    """Separate time series with length 1 from the rest"""
    # get the data
    df['trend_array'] = df[historical_trend_str].apply(extract_popularity_counts)
    # remove columns with YYYY-MM pattern
    date_pattern = re.compile(r'^\d{4}-\d{2}$')
    date_columns = [col for col in df.columns if date_pattern.match(str(col))]
    df = df.drop(columns=date_columns)
    print(f"✅ Created 'trend_array' column and removed date columns.")
    return df

# ------------------------------------------
# create narration column (prep for embedding generation)
# ------------------------------------------
def generate_natural_movie_narration(row):
    """
    Generate natural language narration from movie metadata columns
    """
    def get(col):
        val = row.get(col, '')
        return str(val).strip() if pd.notna(val) and str(val).strip() not in ('', 'NaT') else None

    title    = get('movie_name')
    genre    = get('genre')
    language = get('language')
    director = get('director')
    writers  = get('writers')
    actors   = get('top5_actors')
    producers= get('producers')
    music    = get('music')
    synopsis = get('description')

    # Extract full date
    date_str = None
    release_date = row.get('release_date', '')
    if pd.notna(release_date) and str(release_date).strip() not in ('', 'NaT'):
        try:
            date_str = release_date.strftime('%B %d, %Y')   # → "July 16, 2010"
        except:
            pass

    # --- Build metadata sentence ---
    opening = title or "This film"
    if date_str:
        opening += f" ({date_str})"
    if genre or language:
        lang_str  = f"{language}-language " if language else ""
        genre_str = f"{genre} film" if genre else "film"
        opening  += f" is a {lang_str}{genre_str}"
    opening += "."

    # Crew line: "Directed by X, written by Y, and produced by Z."
    crew_parts = []
    if director:  crew_parts.append(f"directed by {director}")
    if writers:   crew_parts.append(f"written by {writers}")
    if producers: crew_parts.append(f"produced by {producers}")
    crew_line = (crew_parts[0].capitalize() + (
        ", " + ", ".join(crew_parts[1:-1]) + (f", and {crew_parts[-1]}" if len(crew_parts) > 1 else "")
        if len(crew_parts) > 1 else ""
    ) + ".") if crew_parts else None

    # Cast/music line: "Starring A, B, and C, with music by D."
    cast_line = None
    if actors or music:
        cast_line = (f"Starring {actors}" if actors else "") + \
                    (f", with music by {music}" if music else "") + "."
        cast_line = cast_line.lstrip(", ")

    # Synopsis
    synopsis_line = f"Synopsis: {synopsis}" if synopsis else None

    return "\n".join(filter(None, [opening, crew_line, cast_line, synopsis_line]))


def add_narration_column(df):
    """
    Add narration column to dataframe
    """
    print(f"🎬 Generating narration for {len(df)} movies...")
    # df['release_date'] = df['release_date'].dt.date

    df['narration'] = df.apply(generate_natural_movie_narration, axis=1)
    
    # Check empty narrations
    empty_count = df['narration'].apply(lambda x: len(x) == 0).sum()
    print(f"✅ Narration generated: {len(df) - empty_count} movies")
    if empty_count > 0:
        print(f"⚠️ {empty_count} movies have empty narration (missing title & description)")
    return df

# ------------------------------------------
# save based on dataset
# ------------------------------------------
def save_datasets(complete_df, indicator):
    """Save the separated datasets based on the indicator"""
    if indicator == 'validation_set':
        complete_df.to_csv("dataset/260825/260825_full_final_val_movie_data.csv", index=False)
    elif indicator == 'test_set':
        complete_df.to_csv("dataset/260825/260825_full_final_test_movie_data.csv", index=False)
    else:
        complete_df.to_csv("processed_results/sample_corrected_movie_data.csv", index=False)

# ------------------------------------------
# main function
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Inference Data Preprocessing Pipeline')
    parser.add_argument('--dataset', '-d',
    type=str,
    default='validation_set',
    choices=['validation_set', 'test_set', 'sample'],
    help='Dataset selection (default: None)'
    )
    args = parser.parse_args()
    
    # Step 1: Load data
    if args.dataset == 'validation_set':
        # df = data_loading("dataset/processed_complete_val_data.xlsx","test_validation/inter_val_set.csv")
        df = data_loading("dataset/260820/processed_complete_val_data.xlsx")
    elif args.dataset == 'test_set':
        df = data_loading("dataset/260820/processed_complete_test_data.xlsx")
    
    # Step 2: Basic cleaning
    print(df.info())
    df_cleaned = data_cleaning(df,indicator=args.dataset)
    
    # Step 3: Create popularity_month and popularity_value
    df_with_popularity = create_popularity_month_value(df_cleaned, indicator=args.dataset)
    
    # Step 4: Add historical trend column
    df_with_trend = add_historical_trend_column(df_with_popularity)

    # Step 5: Create trend arrays
    complete_df = create_trend_arrays(df_with_trend, 'historical_trend')

    # Step 6: Create item age and latest pop value
    complete_df['item_age'] = complete_df.apply(calculate_item_age, axis=1)
    complete_df['latest_pop_value'] = complete_df['historical_trend'].apply(extract_latest_popularity)
    
    # Step 7: Create narration column and its embeddings
    complete_df = add_narration_column(complete_df)
    
    # # Save results
    save_datasets(complete_df, args.dataset)
    
if __name__ == "__main__":
    main()