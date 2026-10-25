"""
Pointwise util
"""


import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_mse_metric(inferenced_df: pd.DataFrame) -> tuple:
    """
    Compute Mean Squared Error for the given dataframe containing inferenced results.
    
    Mean Squared Error (MSE) is calculated as the average of the squared differences:
    
    $$
    MSE = \\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
    $$
    
    Args:
        inferenced_df (`pd.DataFrame`): DataFrame containing the columns 'ground_truth_pop
    
    Returns:
        (mse, rmse) (`tuple`): A tuple containing the Mean Squared Error and Root Mean Squared Error
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']
    
    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return mse, rmse


def compute_mae_error(inferenced_df: pd.DataFrame) -> float:
    """
    Compute Mean Absolute Error for the given dataframe containing inferenced results.
    
    Mean Absolute Error (MAE) is calculated as the average of the absolute differences:
    
    $$
    MAE = \\frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
    $$
    
    Args:
        inferenced_df (`pd.DataFrame`): DataFrame containing the columns 'ground_truth_popularity_score' and 'predict_popularity_score'.
    
    Returns:
        mae (`float`): The Mean Absolute Error value.
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']
    
    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def compute_r2_score(inferenced_df: pd.DataFrame) -> float:
    """
    Compute R-squared score for the given dataframe containing inferenced results.
    
    R-squared score is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
    
    $$
    R^2 = 1 - \\frac{\\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{N} (y_i - \\bar{y})^2}
    $$
    
    Args:
        inferenced_df (`pd.DataFrame`): DataFrame containing the columns 'ground_truth_popularity_score' and 'predict_popularity_score'.
    Returns:
        r2 (`float`): The R-squared score value.
    """
    y_true = inferenced_df['ground_truth_popularity_score']
    y_pred = inferenced_df['predict_popularity_score']
    
    # if one element is none, set it to 0
    y_true = y_true.fillna(0)
    y_pred = y_pred.fillna(0)
    
    r2 = r2_score(y_true, y_pred)
    return r2


def compute_jaccard_distance(inferenced_df: pd.DataFrame, K=5):
    # Get top-K by predict_popularity_score and ground_truth_popularity_score
    top_k_pred = inferenced_df.sort_values(by='predict_popularity_score', ascending=False).head(K)["movie_id"]
    top_k_true = inferenced_df.sort_values(by='ground_truth_popularity_score', ascending=False).head(K)["movie_id"]

    # Compute Jaccard distance
    intersection = set(top_k_pred).intersection(set(top_k_true))
    union = set(top_k_pred).union(set(top_k_true))
    jaccard_distance = 1 - len(intersection) / len(union) if union else 0

    return jaccard_distance


def compute_kendall_tau_coefficient(inferenced_df: pd.DataFrame):
    pass
