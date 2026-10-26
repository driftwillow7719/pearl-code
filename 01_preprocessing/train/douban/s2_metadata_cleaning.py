import pandas as pd
import re
import ast
from datetime import datetime
from zoneinfo import ZoneInfo
DOUBAN_TZ = ZoneInfo("Asia/Shanghai")

# ==================== DATA LOADING ====================
def data_loading():
    movie_metadata = pd.read_excel('raw_dataset/douban_movie_info_all.xlsx')
    imdb_movie_connector = pd.read_csv("raw_dataset/movie_meta_2w_transed_final.csv")
    train_pop = pd.read_excel('.../processed_popularity_train_data.xlsx')
    val_pop=pd.read_excel('.../processed_popularity_val_data.xlsx')
    test_pop=pd.read_excel('.../processed_popularity_test_data.xlsx')
    return movie_metadata, imdb_movie_connector,train_pop,val_pop,test_pop

# ==================== BASIC CLEANING AND MERGING ====================
def basic_cleaning(movie_metadata, imdb_movie_connector,train_pop,val_pop,test_pop):
    movie_metadata = movie_metadata.drop_duplicates()
    imdb_movie_connector = imdb_movie_connector.drop_duplicates()
    train_pop = train_pop.drop_duplicates()
    val_pop = val_pop.drop_duplicates()
    test_pop = test_pop.drop_duplicates()
    return movie_metadata, imdb_movie_connector,train_pop, val_pop, test_pop

def imdb_movie_connector_cleaning(imdb_movie_connector):
    subset_imdb_connector = imdb_movie_connector[["MovieID", "IMDb","Genre","Description","Language","Release_Date"]]
    subset_imdb_connector.loc[:,'Release_Date'] = subset_imdb_connector['Release_Date'].astype(str)
    
    # check for rows containing letters using the str.contains method with regex
    rows_with_letters = subset_imdb_connector['Release_Date'].str.contains(r'[a-zA-Z]', na=False)
    # filter out rows containing letters
    filtered_meta_subset = subset_imdb_connector.copy()
    filtered_meta_subset = filtered_meta_subset.loc[~rows_with_letters]   
    filtered_meta_subset['Release_Date'] = pd.to_datetime(pd.to_numeric(filtered_meta_subset['Release_Date']),unit='s')
    filtered_meta_subset = filtered_meta_subset.reset_index(drop=True)
    # only keep first data
    filtered_meta_subset = filtered_meta_subset.drop_duplicates(subset='MovieID',keep='first')
    return filtered_meta_subset

def dataset_merging(pop_data,connector,metadata):
    merged_with_connector = pd.merge(pop_data,connector,how='left',on='MovieID')
    # rename columns
    metadata = metadata.rename(columns={'imdb':'IMDb'})
    merged_with_metadata = pd.merge(merged_with_connector,metadata,how='inner',on='IMDb')
    return merged_with_metadata

# ==================== FOR METADATA CLEANING ====================
def get_monthly_cols(end_date):
    # end_date can be '2019-11', '2019-12', or '2020-01'
    dates = pd.date_range(start='2010-01', end=end_date, freq='MS')
    return [d.strftime('%Y-%m') for d in dates]

def reformat_music(data):
    """
    Reformat music entries to extract composer names while preserving all relevant data.
    """
    # Step 1: Split the data by newlines
    lines = data.split('\n')

    # Step 2: Process each line to extract names
    formatted_names = []
    for line in lines:
        # Check for '...' and handle the line accordingly
        if '...' in line:
            name = line.split(' ... ')[0].strip()  # Get the name part before "..."
            formatted_names.append(name)  # Add name to list
        else:
            # If there is no '...', add the entire line to the list
            formatted_names.append(line.strip())  # Retain all names
            
    # Step 3: Join the formatted names with a ", "
    result = ', '.join(formatted_names)
    
    return result

def replace_newline(data):
    return data.replace('\n', ', ')

def format_entry(data):
    """
    Reformat producer entries to convert '...' to 'as' and remove text within parentheses.

    """
    # Step 1: Split the data by newlines
    lines = data.split('\n')
    
    # Step 2: Process each line to format names and remove unwanted text
    formatted_lines = []
    for line in lines:
        # Remove text within parentheses (including parentheses)
        line = re.sub(r"\(.*?\)", '', line)  # Remove text in parentheses
        
        # Replace ' ... ' with ' as '
        line = line.replace(' ... ', ' as ')
        
        # Strip any extra whitespace and add the formatted line to the list
        formatted_lines.append(line.strip())
    
    # Step 3: Join the formatted lines with a ", "
    result = ', '.join(formatted_lines)
    return result

def convert_string_to_list(string_data):
    if isinstance(string_data, str):
        try:
            # Use ast.literal_eval to safely evaluate string as list
            return ast.literal_eval(string_data)
        except (ValueError, SyntaxError):
            return string_data  # return as-is if there's an error
    return string_data  # return as-is if it's not a string

def remove_brackets(row):
    if isinstance(row, list):  # check if it's a list
        capitalized_content = [item.capitalize() for item in row]
        return ', '.join(capitalized_content)  # Convert list to string
    return row  # Return as-is if not a list

# ================= INCLUSION OF POPULARITY RELATED FEATURES =========
def get_release_cutoff(indicator, target_month):
    """
    target_month: the inference/target month for this split, as 'YYYY-MM'.
    test_set  -> cutoff is the target month itself (inclusive), so items
                 released ON the inference month are kept as cold-start rows.
    train/val -> cutoff is one month BEFORE the target month, so items
                 released in the inference month are excluded (they're the
                 thing being predicted, not something the model should have
                 already seen as "existing" at feature-build time).
    """
    target = pd.Period(target_month, freq='M')
    if indicator == 'test_set':
        cutoff = target
    else:  # train_set, validation_set
        cutoff = target - 1
    return str(cutoff)
# ==================== AND METADATA CLEANING PIPELINE ====================    
def metadata_cleaning(merged_df,indicator):
    # remove irrelevant columns
    merged_df = merged_df.drop(columns=['d_title','a_title','w_title','p_title','music_title',
                                                                    'top5_actor_links','director_links','writer_links','all_actors'])
    # remove NaN data rows
    print("before drop na", merged_df.count())
    merged_df = merged_df.dropna()
    # remove duplicates
    merged_df = merged_df.drop_duplicates()
    print("after drop na", merged_df.count())

    # column name reordering
    column_order=['MovieID', 'IMDb', 'movie_name','Description','Release_Date','Genre','Language', 'director',
    'writers', 'top5_actors', 'producers', 'music']
    if indicator=='train_set':
        monthly_columns = get_monthly_cols('2019-10')
        cutoff_month = max(monthly_columns)
    elif indicator=='validation_set':
        monthly_columns = get_monthly_cols('2019-09')
        cutoff_month = get_release_cutoff(indicator, '2019-10')
    elif indicator=='test_set':
        monthly_columns = get_monthly_cols('2019-10')
        cutoff_month = get_release_cutoff(indicator, '2019-11')
        
    release_month = merged_df['Release_Date'].dt.strftime('%Y-%m')
    merged_df = merged_df[release_month <= cutoff_month].copy()

    column_order += monthly_columns
    sorted_merged_df = merged_df[column_order]
    
    # change column name format
    sorted_merged_df.columns = sorted_merged_df.columns.str.lower()
    sorted_merged_df['release_date'] = sorted_merged_df['release_date'].dt.tz_localize(None)

    # transform music column
    sorted_merged_df['music'] = sorted_merged_df['music'].apply(reformat_music)
    
    # transform director, writers, top5_actors
    column_names = ["director", "writers", "top5_actors"]
    for col in column_names:
        sorted_merged_df[col]=sorted_merged_df[col].apply(replace_newline)
        
    # transform producer
    sorted_merged_df['producers'] = sorted_merged_df['producers'].apply(format_entry)
    
    # transform language and genre
    column_names = ['genre','language']
    for col in column_names:
        sorted_merged_df[col]= sorted_merged_df[col].apply(convert_string_to_list)
        sorted_merged_df[col]= sorted_merged_df[col].apply(remove_brackets)

    
    return sorted_merged_df

# ==================== DATA SAVING ====================    
def save_datasets(df,indicator):
    if indicator == 'train_set':
        df.to_excel("dataset/q3/sgt/processed_complete_train_data.xlsx",index=False)
    elif indicator == 'validation_set':
        df.to_excel("dataset/q3/sgt/processed_complete_val_data.xlsx",index=False)
    elif indicator == 'test_set':
        df.to_excel("dataset/q3/sgt/processed_complete_test_data.xlsx",index=False)
    else:
        print("Wrong input, try again")
    print(f"{indicator} cleaned data saved")
    
    
# ==================== FINAL PIPELINE ====================    
def main():
    # load datasets
    movie_metadata, imdb_movie_connector,train_pop, val_pop, test_pop = data_loading()
    
    # basic cleaning
    movie_metadata, imdb_movie_connector,train_pop, val_pop, test_pop = basic_cleaning(movie_metadata, imdb_movie_connector,train_pop,val_pop,test_pop)
    
    # cleaning connector data
    cleaned_connector = imdb_movie_connector_cleaning(imdb_movie_connector)
    
    # merging dataset
    merged_train = dataset_merging(train_pop,cleaned_connector,movie_metadata)
    merged_val = dataset_merging(val_pop,cleaned_connector,movie_metadata)
    merged_test = dataset_merging(test_pop,cleaned_connector,movie_metadata)
    
    print(merged_val.info())
    print(merged_val.columns)
    
    # metadata cleaning
    cleaned_train = metadata_cleaning(merged_train,'train_set')
    cleaned_val = metadata_cleaning(merged_val,'validation_set')
    cleaned_test = metadata_cleaning(merged_test,'test_set')
    
    # only get data with ground truths
    print("Train Set: ",cleaned_train.shape)
    print("Validation Set: ",cleaned_val.shape)
    print("Test Set: ",cleaned_test.shape)
    
    # save data
    save_datasets(cleaned_train,"train_set")
    save_datasets(cleaned_val,"validation_set")
    save_datasets(cleaned_test,"test_set")
    
    print(f"Test set has {len(cleaned_test)} cleaned rows. Previously: {len(merged_test)}")

if __name__ == "__main__":
    main()
    
    
    
    

    
    
    
