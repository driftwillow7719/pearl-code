import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

DOUBAN_TZ = ZoneInfo("Asia/Shanghai")

# data loading
def prepare_dataset():
    dataset = pd.read_csv("raw_dataset/douban_movie_review.csv")
    meta_dataset = pd.read_csv("raw_dataset/movie_meta_2w_transed_final.csv")
    return dataset,meta_dataset

# data cleaning
def clean_dataset(dataset):
    # dropping irrelevant columns
    dataset = dataset.drop(columns=['Unnamed: 0'])

    # sorting and add Date
    dataset_sorted = dataset.sort_values(by='Time',ascending=False).reset_index(drop=True)
    dataset_added_date = dataset_sorted.copy()

    # Keep to DOUBAN_TZ
    dataset_added_date['Curr_Date'] = (
        pd.to_datetime(dataset_added_date['Time'], unit='s', utc=True)
        .dt.tz_convert(DOUBAN_TZ)
    )
    return dataset_added_date

# dataset range
def show_date_range(data):
    min_dataset_added_date_sorted = min(data.Curr_Date)
    max_dataset_added_date_sorted = max(data.Curr_Date)
    print(f"Date range is from {min_dataset_added_date_sorted} to {max_dataset_added_date_sorted}")

def clean_metadata(meta_dataset):
    meta_dataset = meta_dataset.drop(columns=['Unnamed: 0.2','Unnamed: 0','Unnamed: 0.1'])
    meta_subset = meta_dataset[['MovieID','Release_Date']]
    meta_subset.loc[:,'Release_Date'] = meta_subset['Release_Date'].astype(str)

    # check for rows containing letters using the str.contains method with regex
    rows_with_letters = meta_subset['Release_Date'].str.contains(r'[a-zA-Z]', na=False)
    # filter out rows containing letters
    filtered_meta_subset = meta_subset.copy()
    filtered_meta_subset = filtered_meta_subset.loc[~rows_with_letters]
    filtered_meta_subset['Release_Date'] = pd.to_datetime(pd.to_numeric(filtered_meta_subset['Release_Date']),unit='s')

    # create release month
    filtered_meta_subset_month = filtered_meta_subset.copy()
    filtered_meta_subset_month['Release_Month'] = filtered_meta_subset_month['Release_Date'].dt.strftime('%Y-%m')
    # exclude the Release_Date column
    filtered_meta_subset_month = filtered_meta_subset_month.drop(columns='Release_Date')
    filtered_meta_subset_month = filtered_meta_subset_month.reset_index(drop=True)
    # drop duplicated month options
    filtered_meta_subset_unique=filtered_meta_subset_month.drop_duplicates()
    return filtered_meta_subset_unique

def _unix_timestamp(date_str):
    """Parse a 'YYYY-MM-DD' string as DOUBAN_TZ midnight and return its Unix
    epoch seconds, pinned explicitly to Asia/Shanghai instead of relying on the
    running machine's local timezone."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=DOUBAN_TZ).timestamp()

def dataset_cut(dataset):
    # Filter datasets according to date boundaries
    start_date_train = '2010-01-01'
    end_date_train = '2019-11-01'
    start_date_val = '2019-10-01'
    end_date_val = '2019-11-01'
    end_date_test = '2019-12-01'

    ts_start_train = _unix_timestamp(start_date_train)
    ts_end_train = _unix_timestamp(end_date_train)
    ts_start_val = _unix_timestamp(start_date_val)
    ts_end_val = _unix_timestamp(end_date_val)
    ts_end_test = _unix_timestamp(end_date_test)
    
    train_subset = dataset[(dataset['Time'] >= ts_start_train) & (dataset['Time'] < ts_end_train)]
    validation_subset = dataset[(dataset['Time'] >= ts_start_train) & (dataset['Time'] < ts_end_val)]
    validation_ground_truth = dataset[(dataset['Time'] >= ts_start_val) & (dataset['Time'] < ts_end_val)]
    test_subset = dataset[(dataset['Time'] >= ts_start_train) & (dataset['Time'] < ts_end_test)]
    # Nov 2019 target interactions: [2019-11-01 00:00 DOUBAN_TZ, 2019-12-01 00:00 DOUBAN_TZ)
    test_ground_truth = dataset[(dataset['Time'] >= ts_end_val) & (dataset['Time'] < ts_end_test)]
    return train_subset, validation_subset, test_subset, validation_ground_truth, test_ground_truth

def dataset_transformation(dataset):
    # add month data (Curr_Date is now DOUBAN_TZ-aware, so this bucketing is
    # consistent with the DOUBAN_TZ cutoffs used in dataset_cut)
    data_month = dataset.copy()
    data_month['Month'] = data_month['Curr_Date'].dt.strftime('%Y-%m')
    data_month = data_month.reset_index(drop=True)
    # make aggregation
    data_month = data_month.groupby(['MovieID','Month'])['UserID'].count().reset_index()
    return data_month

def data_metadata_merging(dataset,metadata):
    merged_dataset = pd.merge(dataset,metadata[["MovieID","Release_Month"]],on="MovieID")
    return merged_dataset

def merged_data_transformation(merged_df):
    pivot_df = merged_df.pivot(index="MovieID", columns='Month', values="UserID").fillna(0)

    # Step 2: Create a mask based on Month and Release_Month
    for movie_id in pivot_df.index:
        release_month = merged_df.loc[merged_df['MovieID'] == movie_id, 'Release_Month'].values[0]

        for month in pivot_df.columns:
            # Check if Month is before or after Release_Month
            if month < release_month:
                pivot_df.at[movie_id, month] = -1
            elif month >= release_month and pivot_df.at[movie_id, month] == 0:
                pivot_df.at[movie_id, month] = 0
    return pivot_df


def save_datasets(pivot_df,indicator):
    # get MovieID as a column name too
    pivot_df = pivot_df.reset_index()
    if indicator == 'train_set':
        pivot_df.to_excel("popularity_inputs/q3/sgt/processed_popularity_train_data.xlsx",index=False)
    elif indicator == 'validation_set':
        pivot_df = pivot_df.drop(columns=['2019-10']).copy()
        pivot_df.to_excel("popularity_inputs/q3/sgt/processed_popularity_val_data.xlsx",index=False)
    elif indicator == 'test_set':
        pivot_df = pivot_df.drop(columns=['2019-11']).copy()
        pivot_df.to_excel("popularity_inputs/q3/sgt/processed_popularity_test_data.xlsx",index=False)
    else:
        print("Wrong input, try again")
    print(f"{indicator} data saved")

def main():
    dataset,meta_dataset = prepare_dataset()
    cleaned_dataset = clean_dataset(dataset)
    show_date_range(cleaned_dataset)
    cleaned_metadata = clean_metadata(meta_dataset)
    train_data, val_data, test_data, validation_ground_truth, test_ground_truth = dataset_cut(cleaned_dataset)
    show_date_range(train_data)
    show_date_range(val_data)
    show_date_range(validation_ground_truth)
    show_date_range(test_ground_truth)

    # dataset transformation
    transformed_train = dataset_transformation(train_data)
    transformed_val = dataset_transformation(val_data)
    transformed_test = dataset_transformation(test_data)

    # dataset metadata merging
    merged_train = data_metadata_merging(transformed_train,cleaned_metadata)
    merged_test = data_metadata_merging(transformed_test,cleaned_metadata)
    merged_val = data_metadata_merging(transformed_val,cleaned_metadata)

    # final transformation (monthly popularity counts)
    final_train = merged_data_transformation(merged_train)
    final_test = merged_data_transformation(merged_test)
    final_val = merged_data_transformation(merged_val)

    # dataset saving
    save_datasets(final_train,"train_set")
    save_datasets(final_val,"validation_set")
    save_datasets(final_test,"test_set")
    
    # print("Train Data Size: ", final_train.shape)
    validation_ground_truth.to_csv("test_validation/q3/sgt/inter_val_set.csv",index=False)
    test_ground_truth.to_csv("test_validation/q3/sgt/inter_test_set.csv",index=False)

# run
if __name__ == "__main__":
    main()
