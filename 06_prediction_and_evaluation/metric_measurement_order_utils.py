"""
Order related utility functions for metric measurement in recommendation tasks.

Date: 2025-06-27
"""


from typing import List, Dict
import pandas as pd
import numpy as np


# def compute_ndcg_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
#     """
#     Compute the Normalized Discounted Cumulative Gain (NDCG) at K without user.
    
#     The relevant measurement is based on `ground_truth_score` and it will be normalized.
    
#     Args:~
#         inferenced_df (pd.DataFrame): DataFrame containing the inference results.
#         K (int): The cutoff rank for NDCG calculation.
    
#     Returns:
#         float: The NDCG value at K.
#     """
#     inferenced_df = inferenced_df.copy()

    
#     # inferenced_df["relevant"] = (inferenced_df["Rate"] >= 3).astype(int)
#     inferenced_df["relevant"] = inferenced_df["Rate"]
#     if inferenced_df["relevant"].sum() == 0:
#         return 0.0
    
#     # sort
#     sorted_df = inferenced_df.sort_values(by='predict_popularity_score', ascending=False)
#     top_k_relevance = sorted_df["relevant"].values[:K]
    
#     # calculate DCG@K
#     dcg_at_k = np.sum(top_k_relevance / np.log2(np.arange(2, len(top_k_relevance) + 2)))
    
#     # calculate IDCG@K
#     ideal_relevance = np.sort(inferenced_df["relevant"].values)[::-1][:K]
#     idcg_at_k = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
#     ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
#     return ndcg_at_k

def compute_ndcg_at_k(inferenced_df: pd.DataFrame,
                      user_interact_df: pd.DataFrame,
                      K: int,
                      score_col: str = "predict_popularity_score") -> float:
    """
    NDCG@K for ONE user, same formula as TRAIL:
      - binary relevance (Rate > 0)
      - IDCG over all of the user's relevant items in the window
      - final value divided by the number of the user's interaction rows

    inferenced_df    : the user's ranked top-k list (movieid + score columns)
    user_interact_df : ONLY this user's interactions in the evaluation window
    score_col        : column that defines the ranking
    """
    inf = inferenced_df.copy()
    usr = user_interact_df.copy()

    # unify column names (Douban: MovieID/Rate, Amazon: item/rating)
    if "movieid" not in usr.columns:
        for c in ("MovieID", "item", "item_id"):
            if c in usr.columns:
                usr = usr.rename(columns={c: "movieid"})
                break
    if "Rate" not in usr.columns and "rating" in usr.columns:
        usr = usr.rename(columns={"rating": "Rate"})
    if "movieid" not in inf.columns and "item_id" in inf.columns:
        inf = inf.rename(columns={"item_id": "movieid"})

    if len(usr) == 0:
        return 0.0

    # align id dtypes: int if possible (matches TRAIL), otherwise string
    try:
        inf["movieid"] = inf["movieid"].astype(int)
        usr["movieid"] = usr["movieid"].astype(int)
    except (ValueError, TypeError):
        inf["movieid"] = inf["movieid"].astype(str).str.strip()
        usr["movieid"] = usr["movieid"].astype(str).str.strip()

    df = pd.merge(inf[["movieid", score_col]], usr[["movieid", "Rate"]],
                  on="movieid", how="outer").reset_index(drop=True)
    df[score_col] = df[score_col].fillna(-np.inf)
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(0.0)
    rel = (df["Rate"] > 0).astype(float).to_numpy()

    discounts = np.log2(np.arange(2, K + 2))

    # DCG: rank by score (ties by movieid)
    order = df.sort_values(by=[score_col, "movieid"], ascending=[False, True]).index
    rel_dcg = rel[order][:K]
    dcg = float(np.sum(rel_dcg / discounts[:len(rel_dcg)]))

    # IDCG: all relevant items of this user first
    order_i = df.sort_values(by=["Rate", "movieid"], ascending=[False, True]).index
    rel_idcg = (df.loc[order_i, "Rate"] > 0).astype(float).to_numpy()[:K]
    idcg = float(np.sum(rel_idcg / discounts[:len(rel_idcg)]))

    if idcg == 0.0:
        return 0.0
    return (dcg / idcg) / len(usr)

def compute_map_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    """
    Compute MAP@K (Mean Average Precision at K) without user.
    
    The relevant measurement is based on `ground_truth_popularity_score`.
    
    Args:
        inferenced_df (pd.DataFrame): DataFrame containing the inference results.
        K (int): The cutoff rank for MAP calculation.
    
    Returns:
        float: The MAP value at K.
    """
    inferenced_df = inferenced_df.copy()
    
    inferenced_df["relevant"] = (inferenced_df["Rate"] > 0).astype(int)
    
    # sort by model predictions
    sorted_df = inferenced_df.sort_values(by="predict_popularity_score", ascending=False).head(K)
    rels = sorted_df["relevant"].values
    
    precisions = []
    num_relevant = 0
    
    for i in range(len(rels)):
        if rels[i] == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    map_k = np.mean(precisions) if precisions else 0.0
    return map_k


def compute_mrr_at_k(inferenced_df: pd.DataFrame, K: int) -> float:
    """
    Compute MRR@K without user.
    The relevant measurement is based on `ground_truth_popularity_score`.
    
    Args:
        inferenced_df (pd.DataFrame): DataFrame containing the inference results.
        K (int): The cutoff rank for MRR calculation.
    
    Returns:
        float: The MRR value at K.
    """
    inferenced_df = inferenced_df.copy()
    
    inferenced_df["relevant"] = (inferenced_df["Rate"] > 0).astype(int)
    sorted_df = inferenced_df.sort_values(by="predict_popularity_score", ascending=False).head(K)
    relevance_list = sorted_df["relevant"].values
    
    for idx, rel in enumerate(relevance_list):
        if rel == 1:
            mrr_at_k = 1 / (idx + 1)
            return mrr_at_k
    
    return 0.0  # If no relevant item is found in the top K results


if __name__ == "__main__":
    import sys
    movie_df = pd.read_csv(sys.argv[1], nrows=1000)  # usage: python metric_measurement_order_utils.py <csv>
    
    print(movie_df.head())
    
    print("NDCG@20:", compute_ndcg_at_k(movie_df, K=20))
    print("MAP@20:", compute_map_at_k(movie_df, K=20))
    print("MRR@20:", compute_mrr_at_k(movie_df, K=20))