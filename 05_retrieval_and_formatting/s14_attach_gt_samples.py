"""
Attaching ground truth (Douban)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_clustering_and_indexing"))

from refined_retrieval_utils import (
    attach_ground_truth_direction
)
from s5_save_eight_groups import get_history_depth_bucket

def inf_data_loading(inf_data_path,inf_embed_data_path):
    df = pd.read_csv(inf_data_path)
    embed = np.load(
        inf_embed_data_path,
        mmap_mode='r'
    )

    df['trend_array'] = df['trend_array'].apply(
        lambda x: np.array([int(i) for i in x.strip('[]').split()]) if isinstance(x, str) else x
    )
    df['doc_id'] = [i for i in range(len(df))]

    # df['age_cluster'] = df['item_age'].apply(get_age_bucket)    
    # known_mask = df['latest_pop_value'].notna()     
    # df.loc[known_mask, 'item_age'] = df.loc[known_mask, 'item_age'] + 1
    df['history_depth_bucket'] = df.apply(lambda row: get_history_depth_bucket(row['item_age']),axis=1)   
    df = df.rename(columns={'movieid':'movie_id'}) 

    subset_df = df.copy()
    return subset_df, embed


def sample_balanced_groups(df, group_cols=('ground_truth_pop_cluster', 'ground_truth_direction'),
                            target_total=100, random_state=42,
                            exclude_debut=True, age_cluster_col='history_depth_bucket'):
    """
    df: dataframe with ground-truth pop_cluster / pop_direction columns
        (and age_cluster, if exclude_debut=True).

    Returns: (sample_df, summary_df)
        sample_df   -- the balanced sample, with a copy of the original index
        summary_df  -- one row per group: available, target, actual taken
    """
    rng = np.random.default_rng(random_state)
    work_df = df.copy()

    if exclude_debut:
        if age_cluster_col not in work_df.columns:
            raise ValueError(
                f"exclude_debut=True but '{age_cluster_col}' not found in df.columns. "
                f"Available columns: {list(work_df.columns)}"
            )
        work_df = work_df[work_df[age_cluster_col] != 'debut'].copy()

    group_cols = list(group_cols)
    groups = {key: g for key, g in work_df.groupby(group_cols)}
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("No groups found -- check group_cols and that df isn't empty after filtering.")

    base_n = target_total // n_groups
    remainder = target_total % n_groups
    group_keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)

    # give the extra +1 (from the remainder) to the LARGEST groups first --
    # they're the ones actually able to absorb it, unlike a small group like
    # high_popular__stable which is already capped well below base_n
    targets = {}
    for i, key in enumerate(group_keys):
        targets[key] = base_n + (1 if i < remainder else 0)

    taken_idx = {}
    shortfall = 0
    for key in group_keys:
        available = len(groups[key])
        want = min(targets[key], available)
        shortfall += targets[key] - want
        if want > 0:
            chosen = rng.choice(groups[key].index.values, size=want, replace=False)
        else:
            chosen = np.array([], dtype=groups[key].index.dtype)
        taken_idx[key] = list(chosen)

    # redistribute shortfall (from capped small groups) across groups that
    # still have spare, untaken capacity -- largest spare capacity first
    if shortfall > 0:
        spare = {
            key: len(groups[key]) - len(taken_idx[key])
            for key in group_keys
        }
        # keep looping until shortfall is resolved or no group has any spare left
        while shortfall > 0 and any(v > 0 for v in spare.values()):
            for key in sorted(spare, key=lambda k: spare[k], reverse=True):
                if shortfall <= 0:
                    break
                if spare[key] <= 0:
                    continue
                remaining_pool = groups[key].index.difference(taken_idx[key])
                extra = rng.choice(remaining_pool.values, size=1, replace=False)
                taken_idx[key].extend(extra)
                spare[key] -= 1
                shortfall -= 1

    sample_df = pd.concat([work_df.loc[taken_idx[key]] for key in group_keys])
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=False)
    sample_df = sample_df.rename(columns={'index': 'original_index'})

    summary_rows = []
    for key in group_keys:
        pop, direction = key
        summary_rows.append({
            'pop_cluster': pop,
            'pop_direction': direction,
            'available': len(groups[key]),
            'target': targets[key],
            'actual_taken': len(taken_idx[key]),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(['pop_cluster', 'pop_direction']).reset_index(drop=True)

    print(f"Requested {target_total}, got {len(sample_df)} total "
          f"across {n_groups} groups (unmet shortfall: {shortfall} -- "
          f"only possible if every group hit capacity, which is unlikely "
          f"unless target_total is very large relative to your dataset).")
    print(summary_df.to_string(index=False))

    return sample_df, summary_df


if __name__ == "__main__":
    # INTER_PATH = ".../inter_test_set.csv"
    INTER_PATH = ".../inter_val_set.csv"
    # TARGET_MONTH = '2019-11'
    TARGET_MONTH = '2019-10'
    TARGET_SAMPLE = 100
    # INF_DATA_PATH= f"260826_test_embed/final_data.csv"
    INF_DATA_PATH= f"260828_val_embed/final_data.csv"
    # INF_EMBED_DATA_PATH = "260826_test_embed/final_embeddings.npy"
    INF_EMBED_DATA_PATH = "260828_val_embed/final_embeddings.npy"

    subset_df, embed = inf_data_loading(INF_DATA_PATH, INF_EMBED_DATA_PATH)
    dataset = attach_ground_truth_direction(subset_df, INTER_PATH, TARGET_MONTH, id_col='movie_id') 
    dataset['ground_truth_popularity_score'] = dataset['ground_truth_popularity_score'].astype("Int64")
    # sample_df, summary_df = sample_balanced_groups(dataset, target_total=TARGET_SAMPLE)
    dataset.to_csv('.../260828_full_val_set.csv', index=False)
