"""
Age-stratified point-biserial correlation of direction against each feature,
run separately within each history-depth bucket (debut / recent_single_point /
recent_multi_point / maturing). Works for every dataset in dataset_config.py.

Features are log1p-transformed the same way composite_score transforms them,
so the correlations are measured in the space the similarity scoring uses.

Input: the completed training CSV and embeddings produced by
s5_save_eight_groups.py. Buckets come from get_history_depth_bucket(item_age).

Run:  python s6_age_stratified_corr.py --dataset douban
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
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


def add_trend_stats(dataset, trend_col='trend_array'):
    stats_df = dataset[trend_col].apply(trend_summary_stats).apply(pd.Series)
    return pd.concat([dataset.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)


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
    """Returns results keyed by (pop_cluster, pop_direction), each entry
    carrying a pooled 'mean_embed' (all non-debut buckets combined)."""
    dataset = _ensure_history_depth_bucket(dataset)
    non_debut = dataset[dataset['history_depth_bucket'] != 'debut'].copy()
    trend_stats_df = non_debut[trend_col].apply(trend_summary_stats).apply(pd.Series)
    non_debut_with_trends = pd.concat(
        [non_debut.reset_index(drop=True), trend_stats_df.reset_index(drop=True)], axis=1
    )
    results = {}
    for (pop, direction), group in non_debut_with_trends.groupby(['pop_cluster', 'pop_direction']):
        doc_indices = group[doc_id_col].values
        mean_embed = np.mean(embed[doc_indices], axis=0)

        results[(pop, direction)] = {
            'doc_ids': doc_indices,
            'n_items': len(group),
            'mean_embed': mean_embed,
        }
    return results, non_debut_with_trends


def add_embedding_similarity(df, embed, mean_embed_by_group, doc_id_col='doc_id',
                              pop_col='pop_cluster', direction_col='pop_direction',
                              bucket_col='history_depth_bucket'):
    """Compares each row's embedding against its OWN (pop_cluster,
    pop_direction) centroid when one exists."""
    def compute_sim(row):
        key = (row[pop_col], row[direction_col])
        if key not in mean_embed_by_group:
            return np.nan
        group_entry = mean_embed_by_group[key]

        if isinstance(group_entry, dict):
            pooled_mean = group_entry.get('mean_embed', group_entry)

        item_embed = embed[row[doc_id_col]]
        return embed_scoring(item_embed, pooled_mean)
    df = df.copy()
    df['embed_similarity'] = df.apply(compute_sim, axis=1)
    return df


def build_debut_rows(dataset, embed, results, trend_col='trend_array'):
    """Debut-bucket items are excluded from build_group_characteristics --
    no real trend history -- but they still have a TRUE (pop_cluster,
    pop_direction) outcome and a real embedding, so embed_similarity can
    still be computed against the 8 non-debut group centroids."""
    dataset = _ensure_history_depth_bucket(dataset)
    debut = dataset[dataset['history_depth_bucket'] == 'debut'].copy()
    debut['latest_pop_value'] = 0
    debut_with_trends = add_trend_stats(debut, trend_col=trend_col)
    debut_df = add_embedding_similarity(debut_with_trends, embed, results)
    print(f"debut items: {len(debut_df)}")
    print(debut_df['pop_direction'].value_counts(dropna=False))  # should never show 'decrease'
    return debut_df


# ---------------------------------------------------------------------------
# feature -> transform, matching composite_score's log1p routing exactly
# ---------------------------------------------------------------------------

NONNEGATIVE_LOG_FEATURES = {
    'latest_pop_value', 'item_age', 'trend_min', 'trend_max',
    'trend_volatility', 'time_since_peak',
}
SIGNED_LOG_FEATURES = {'trend_overall_change'}


def _apply_feature_transform(feat_name, values):
    """values: a numeric pandas Series (post pd.to_numeric, may contain NaN)."""
    if feat_name in NONNEGATIVE_LOG_FEATURES:
        return np.log1p(values)
    if feat_name in SIGNED_LOG_FEATURES:
        return np.sign(values) * np.log1p(np.abs(values))
    return values


# ---------------------------------------------------------------------------
# Age-stratified correlation (unchanged)
# ---------------------------------------------------------------------------

UNIFIED_BUCKET_ORDER = ['debut', 'recent_single_point', 'recent_multi_point', 'maturing']

def age_stratified_correlation_study(
    df,
    pop_cluster,
    direction,
    features,
    age_col='history_depth_bucket',
    pop_col='pop_cluster',
    direction_col='pop_direction',
    age_buckets=('recent_single_point', 'recent_multi_point', 'maturing'),
):
    rows = []
    for bucket in age_buckets:
        subset = df[(df[pop_col] == pop_cluster) & (df[age_col] == bucket)]
        if subset.empty:
            continue
        indicator = (subset[direction_col] == direction).astype(int)
        if indicator.nunique() < 2:
            print(f"Skipping {pop_cluster}/{bucket}: no variation in direction=={direction}")
            continue
        PEAK_SOURCE_FEATURES = {'trend_min', 'trend_max', 'time_since_peak'}

        for feat in features:
            feat_values = pd.to_numeric(subset[feat], errors='coerce')
            feat_values = _apply_feature_transform(feat, feat_values)
            valid = feat_values.notna()
            if feat in PEAK_SOURCE_FEATURES:
                valid = valid & ~subset['is_single_real_trend_point']
            if valid.sum() < 10 or indicator[valid].nunique() < 2:
                r, p, n = float('nan'), float('nan'), int(valid.sum())
            else:
                r, p = pointbiserialr(indicator[valid], feat_values[valid])
                n = int(valid.sum())
            rows.append({
                'pop_cluster': pop_cluster, 'direction': direction,
                'age_bucket': bucket, 'feature': feat, 'r': r, 'p_value': p, 'n': n,
            })
    result = pd.DataFrame(rows, columns=['pop_cluster', 'direction', 'age_bucket', 'feature', 'r', 'p_value', 'n'])
    if result.empty:
        result['abs_r'] = pd.Series(dtype=float)
        return result
    result['abs_r'] = result['r'].abs()
    result['age_bucket'] = pd.Categorical(
        result['age_bucket'], categories=list(age_buckets), ordered=True
    )
    return result.sort_values(['age_bucket', 'abs_r'], ascending=[True, False])

ALL_OTHER_FEATURES = [
    'latest_pop_value', 'item_age', 'trend_min', 'trend_max',
    'trend_volatility', 'trend_overall_change', 'time_since_peak',
    'latest_momentum',
]
EMBED_FEATURE = 'embed_similarity'


def run_all_age_stratified_studies(combined_df,
                                    other_features=ALL_OTHER_FEATURES,
                                    embed_feature=EMBED_FEATURE,
                                    age_col='history_depth_bucket',
                                    pop_col='pop_cluster', direction_col='pop_direction'):
    features_order = [embed_feature] + list(other_features)

    all_results = []
    pop_clusters = combined_df[pop_col].dropna().unique()
    for pop_cluster in pop_clusters:
        directions = combined_df.loc[
            combined_df[pop_col] == pop_cluster, direction_col
        ].dropna().unique()

        for direction in directions:
            print(f"\n--- {pop_cluster} / {direction} ---")

            embed_res = age_stratified_correlation_study(
                combined_df, pop_cluster, direction, [embed_feature],
                age_col=age_col, age_buckets=UNIFIED_BUCKET_ORDER,
            )
            other_res = age_stratified_correlation_study(
                combined_df, pop_cluster, direction, other_features,
                age_col=age_col, age_buckets=UNIFIED_BUCKET_ORDER[1:],
            )

            combined_block = pd.concat([embed_res, other_res], ignore_index=True)
            combined_block['age_bucket'] = pd.Categorical(
                combined_block['age_bucket'].astype(str),
                categories=UNIFIED_BUCKET_ORDER, ordered=True,
            )
            combined_block['feature'] = pd.Categorical(
                combined_block['feature'], categories=features_order, ordered=True
            )
            combined_block = combined_block.sort_values(['feature', 'age_bucket'])

            all_results.append(combined_block)

    return pd.concat(all_results, ignore_index=True)


def debut_correlation_study(combined_df, feature=EMBED_FEATURE,
                             age_col='history_depth_bucket', direction_col='pop_direction'):
    debut_rows = combined_df[combined_df[age_col] == 'debut']
    indicator = (debut_rows[direction_col] == 'increase').astype(int)
    feat_values = pd.to_numeric(debut_rows[feature], errors='coerce')
    feat_values = _apply_feature_transform(feature, feat_values)
    valid = feat_values.notna() & indicator.notna()
    n = int(valid.sum())
    if n < 10 or indicator[valid].nunique() < 2:
        r, p = float('nan'), float('nan')
    else:
        r, p = pointbiserialr(indicator[valid], feat_values[valid])
    return pd.DataFrame([{
        'pop_cluster': '(all debut items)', 'direction': 'increase_vs_stable',
        'age_bucket': 'debut', 'feature': feature, 'r': r, 'p_value': p,
        'n': n, 'abs_r': abs(r) if pd.notna(r) else np.nan,
    }])


def report_bucket_composition(combined_df, pop_col='pop_cluster',
                               direction_col='pop_direction', age_col='history_depth_bucket'):
    counts = (
        combined_df.groupby([pop_col, direction_col, age_col], observed=True)
        .size()
        .reset_index(name='n_true_members')
    )
    print("\n=== TRUE MEMBER COUNTS PER (pop_cluster, direction, age_bucket) ===")
    print(counts.to_string(index=False))
    return counts


def _parse_trend_array_str(x):
    """Re-parses trend_array from its saved-CSV string form, e.g.
    '[1. 1. 2.]' or '[]'. Uses float() (not the movie reference's int()) --
    Amazon values aren't guaranteed integer-valued and float() still parses
    plain integers fine."""
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
    print(dataset.info())
    embed = np.load(embed_npy, mmap_mode='r')
    assert len(dataset) == len(embed), (
        f"row-count mismatch: dataset has {len(dataset)} rows but embed has {len(embed)}"
    )

    dataset['trend_array'] = dataset['trend_array'].apply(_parse_trend_array_str)
    dataset['is_single_real_trend_point'] = dataset['trend_array'].apply(_is_single_real_trend_point)

    # non-debut items + embed_similarity against their own group's centroid
    results, non_debut_with_trends = build_group_characteristics(dataset, embed)
    non_debut_df = add_embedding_similarity(non_debut_with_trends, embed, results)

    # debut items + embed_similarity against the same 8 centroids
    debut_df = build_debut_rows(dataset, embed, results)

    # combine so 'debut' is just another history_depth_bucket value in one dataframe
    combined_df = pd.concat([non_debut_df, debut_df], ignore_index=True, sort=False)

    full_results = run_all_age_stratified_studies(combined_df)
    debut_result = debut_correlation_study(combined_df)
    bucket_composition = report_bucket_composition(combined_df)

    print("\n\n=== FULL AGE-STRATIFIED CORRELATION STUDY ===")
    print(full_results.to_string(index=False))
    print("\n=== DEBUT (pooled, increase vs. stable) ===")
    print(debut_result.to_string(index=False))

    full_results.to_csv(f"{out_dir}/{name}_age_stratified_correlation_full.csv", index=False)
    bucket_composition.to_csv(f"{out_dir}/{name}_bucket_composition.csv", index=False)
    with open(f"{out_dir}/{name}_age_stratified_correlation_output.txt", "w") as f:
        f.write(full_results.to_string(index=False))
        f.write("\n\n=== TRUE MEMBER COUNTS PER (pop_cluster, direction, age_bucket) ===\n")
        f.write(bucket_composition.to_string(index=False))
    print(f"\nSaved to {out_dir}/{name}_age_stratified_correlation_full.csv, "
        f"{name}_bucket_composition.csv, and "
        f"{name}_age_stratified_correlation_output.txt")
