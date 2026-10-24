"""
Pearson correlation between item_age and every other feature, computed
separately within each of the 8 pop_cluster x direction groups, in the same
log1p space composite_score uses. Works for every dataset in dataset_config.py.

Input: the completed training CSV and embeddings produced by
s5_save_eight_groups.py. Debut items are excluded (item_age is constant there).

Run:  python s7_item_age_correlation_study.py --dataset douban
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from s5_save_eight_groups import get_history_depth_bucket, _is_single_real_trend_point
from dataset_config import DATASETS

# ---------------------------------------------------------------------------
# Shared helpers (unchanged from v5)
# ---------------------------------------------------------------------------

def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unavailable"
    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w+1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def trend_summary_stats(trend_array):
    arr = np.asarray(trend_array, dtype=float) if isinstance(trend_array, (list, np.ndarray)) else None
    if arr is None or len(arr) == 0 or (len(arr) == 1 and arr[0] == -1):
        return {
            'trend_min': np.nan, 'trend_max': np.nan, 'trend_volatility': np.nan,
            'trend_overall_change': np.nan, 'time_since_peak': np.nan,
        }
    trend_min = float(arr.min())
    trend_max = float(arr.max())
    trend_volatility = float(arr.std()) if len(arr) > 1 else np.nan
    trend_overall_change = float(arr[-1] - arr[0]) if len(arr) > 1 else np.nan
    peak_idx = int(arr.argmax())
    time_since_peak = (len(arr) - 1) - peak_idx
    return {
        'trend_min': trend_min, 'trend_max': trend_max,
        'trend_volatility': trend_volatility,
        'trend_overall_change': trend_overall_change,
        'time_since_peak': time_since_peak,
    }


def embed_scoring(embed, center_vector):
    distance = cosine(embed, center_vector)
    similarity = 1 - distance
    return (1 + similarity) / 2


def _ensure_history_depth_bucket(dataset):
    if 'history_depth_bucket' not in dataset.columns:
        dataset = dataset.copy()
        dataset['history_depth_bucket'] = dataset.apply(
            lambda row: get_history_depth_bucket(row['item_age']), axis=1
        )
    return dataset


def build_group_characteristics(dataset, embed, doc_id_col='doc_id', trend_col='trend_array'):
    dataset = _ensure_history_depth_bucket(dataset)
    non_debut = dataset[dataset['history_depth_bucket'] != 'debut'].copy()
    trend_stats_df = non_debut[trend_col].apply(trend_summary_stats).apply(pd.Series)
    non_debut_with_trends = pd.concat(
        [non_debut.reset_index(drop=True), trend_stats_df.reset_index(drop=True)], axis=1
    )
    results = {}
    for (pop, direction), group in non_debut_with_trends.groupby(['pop_cluster', 'pop_direction']):
        doc_indices = group[doc_id_col].values
        embeddings = embed[doc_indices]
        mean_embed = np.mean(embeddings, axis=0)
        results[(pop, direction)] = {
            'doc_ids': doc_indices,
            'n_items': len(group),
            'mean_embed': mean_embed,
        }
    return results, non_debut_with_trends


def add_embedding_similarity(df, embed, mean_embed_by_group, doc_id_col='doc_id',
                              pop_col='pop_cluster', direction_col='pop_direction'):
    def compute_sim(row):
        key = (row[pop_col], row[direction_col])
        if key not in mean_embed_by_group:
            return np.nan
        group_mean = mean_embed_by_group[key]
        if isinstance(group_mean, dict):
            group_mean = group_mean['mean_embed']
        item_embed = embed[row[doc_id_col]]
        return embed_scoring(item_embed, group_mean)
    df = df.copy()
    df['embed_similarity'] = df.apply(compute_sim, axis=1)
    return df


# ---------------------------------------------------------------------------
# feature -> transform, matching composite_score's log1p routing exactly
# ---------------------------------------------------------------------------

NONNEGATIVE_LOG_FEATURES = {
    'item_age', 'latest_pop_value', 'trend_min', 'trend_max',
    'trend_volatility', 'time_since_peak',
}
SIGNED_LOG_FEATURES = {'trend_overall_change'}
# deliberately untransformed: 'embed_similarity' (cosine-based),
# 'latest_momentum' (already an average log1p-diff internally)


def _apply_feature_transform(feat_name, values):
    """values: a numeric pandas Series (post pd.to_numeric, may contain NaN)."""
    if feat_name in NONNEGATIVE_LOG_FEATURES:
        return np.log1p(values)
    if feat_name in SIGNED_LOG_FEATURES:
        return np.sign(values) * np.log1p(np.abs(values))
    return values


# ---------------------------------------------------------------------------
# The actual test: item_age vs. every other feature, within each group
# ---------------------------------------------------------------------------

FEATURES_TO_TEST = [
    'latest_pop_value', 'trend_min', 'trend_max', 'trend_volatility',
    'trend_overall_change', 'time_since_peak', 'latest_momentum',
    'embed_similarity',
]

def item_age_correlation_study(df, features=FEATURES_TO_TEST,
                                pop_col='pop_cluster', direction_col='pop_direction',
                                age_col='item_age'):
    PEAK_SOURCE_FEATURES = {'trend_min', 'trend_max', 'time_since_peak'}
    df = df.copy()
    df['is_single_real_trend_point'] = df['trend_array'].apply(_is_single_real_trend_point)

    rows = []
    for (pop, direction), group in df.groupby([pop_col, direction_col]):
        age_values = pd.to_numeric(group[age_col], errors='coerce')
        age_values = _apply_feature_transform(age_col, age_values)
        for feat in features:
            feat_values = pd.to_numeric(group[feat], errors='coerce')
            feat_values = _apply_feature_transform(feat, feat_values)
            valid = pd.concat([age_values.rename('age'), feat_values.rename(feat)], axis=1).dropna()
            if feat in PEAK_SOURCE_FEATURES:
                valid = valid[~group.loc[valid.index, 'is_single_real_trend_point']]
            n = len(valid)
            if n < 10 or valid['age'].nunique() < 2 or valid[feat].nunique() < 2:
                r, p = float('nan'), float('nan')
            else:
                r, p = pearsonr(valid['age'], valid[feat])
            rows.append({
                'pop_cluster': pop, 'direction': direction, 'feature': feat,
                'r': r, 'p_value': p, 'n': n,
            })
    result = pd.DataFrame(rows)
    result['abs_r'] = result['r'].abs()
    return result.sort_values(['pop_cluster', 'direction', 'abs_r'], ascending=[True, True, False])


def _parse_trend_array_str(x):
    if not isinstance(x, str):
        return x
    s = x.strip('[]').split()
    return np.array([float(i) for i in s])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--completed-csv', default=None)
    parser.add_argument('--embed-npy', default=None)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    name = args.dataset
    completed_csv = args.completed_csv or f"{cfg['out_dir']}/{name}_completed_train_data.csv"
    embed_npy = args.embed_npy or cfg["embed"]
    out_dir = args.out_dir or f"{cfg['out_dir']}/corr_data"
    os.makedirs(out_dir, exist_ok=True)

    dataset = pd.read_csv(completed_csv).reset_index(drop=True)
    embed = np.load(embed_npy, mmap_mode='r')
    assert len(dataset) == len(embed), (
        f"row-count mismatch: dataset has {len(dataset)} rows but "
        f"embed has {len(embed)} -- doc_id positions won't line up with embed rows"
    )

    dataset['trend_array'] = dataset['trend_array'].apply(_parse_trend_array_str)

    results, non_debut_with_trends = build_group_characteristics(dataset, embed)
    non_debut_df = add_embedding_similarity(non_debut_with_trends, embed, results)

    # debut excluded entirely -- item_age is a constant -1 there, no
    # variance to correlate against
    item_age_results = item_age_correlation_study(non_debut_df)

    print("\n=== ITEM_AGE vs. ALL FEATURES, WITHIN EACH OF THE 8 GROUPS (log1p space) ===")
    print(item_age_results.to_string(index=False))

    item_age_results.to_csv(f"{out_dir}/{name}_item_age_correlation_study.csv", index=False)
    with open(f"{out_dir}/{name}_item_age_correlation_study.txt", "w") as f:
        f.write(item_age_results.to_string(index=False))
    print(f"\nSaved to {out_dir}/{name}_item_age_correlation_study.csv and "
        f"{name}_item_age_correlation_study.txt")
