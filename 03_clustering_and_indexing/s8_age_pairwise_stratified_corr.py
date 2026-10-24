"""
Pairwise group-separability test: for chosen pairs of (pop_cluster, direction)
groups, tests within each history-depth bucket whether each feature separates
group A from group B (point-biserial, in the same log1p space as
composite_score). Works for every dataset in dataset_config.py.

The pairs to compare are dataset-specific (see PAIRS below): they should be
the groups that the s6/s7 results show to be hard to tell apart for that
dataset. Datasets with no pairs listed are skipped with a message.

Input: the completed training CSV and embeddings produced by
s5_save_eight_groups.py.

Run:  python s8_age_pairwise_stratified_corr.py --dataset home_and_kitchen
"""

import argparse
import os

import numpy as np
import pandas as pd
import contextlib
from scipy.stats import pointbiserialr
from s6_age_stratified_corr import (
    build_group_characteristics,
    add_embedding_similarity,
    build_debut_rows,
    run_all_age_stratified_studies,
    debut_correlation_study,
    report_bucket_composition)
from s5_save_eight_groups import _is_single_real_trend_point
from dataset_config import DATASETS


# ---------------------------------------------------------------------------
# feature -> transform, matching composite_score's log1p routing exactly
# ---------------------------------------------------------------------------

NONNEGATIVE_LOG_FEATURES = {
    'latest_pop_value', 'item_age', 'trend_min', 'trend_max',
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


def pairwise_group_separability_stratified(
    df, group_a, group_b, characteristics,
    pop_col='pop_cluster', direction_col='pop_direction',
    age_col='history_depth_bucket',
    age_buckets=('recent_single_point', 'recent_multi_point', 'maturing'),
):
    label_a = f"{group_a[0]}__{group_a[1]}"
    label_b = f"{group_b[0]}__{group_b[1]}"

    PEAK_SOURCE_FEATURES = {'trend_min', 'trend_max', 'time_since_peak'}

    sub_all = df[
        ((df[pop_col] == group_a[0]) & (df[direction_col] == group_a[1])) |
        ((df[pop_col] == group_b[0]) & (df[direction_col] == group_b[1]))
    ].copy()
    sub_all['_is_group_b'] = (
        (sub_all[pop_col] == group_b[0]) & (sub_all[direction_col] == group_b[1])
    ).astype(int)

    rows = []
    for bucket in age_buckets:
        sub = sub_all[sub_all[age_col] == bucket]
        n_a = int((sub['_is_group_b'] == 0).sum())
        n_b = int((sub['_is_group_b'] == 1).sum())
        if n_a == 0 or n_b == 0:
            print(f"  [{bucket}] skipped -- one side has zero members here "
                  f"({label_a}: n={n_a}, {label_b}: n={n_b})")
            continue

        for feat in characteristics:
            if feat not in sub.columns:
                continue
            feat_numeric = pd.to_numeric(sub[feat], errors='coerce')
            feat_numeric = _apply_feature_transform(feat, feat_numeric)
            valid = pd.concat([feat_numeric.rename(feat), sub['_is_group_b']], axis=1).dropna()
            if feat in PEAK_SOURCE_FEATURES:
                valid = valid[~sub.loc[valid.index, 'is_single_real_trend_point']]
            n = len(valid)
            if n < 3 or valid[feat].nunique() < 2 or valid['_is_group_b'].nunique() < 2:
                rows.append({'age_bucket': bucket, 'feature': feat, 'correlation': np.nan,
                             'p_value': np.nan, 'n': n, 'note': 'insufficient data/variation'})
                continue
            corr, p_value = pointbiserialr(valid['_is_group_b'], valid[feat])
            if pd.isna(corr):
                note = 'undefined -- no variance in one variable'
            elif n < 30:
                note = 'SMALL SAMPLE -- exploratory only'
            elif p_value >= 0.05:
                note = 'not significant at p<0.05'
            else:
                note = 'significant'
            rows.append({'age_bucket': bucket, 'feature': feat,
                         'correlation': round(corr, 4) if pd.notna(corr) else np.nan,
                         'p_value': round(p_value, 4) if pd.notna(p_value) else np.nan,
                         'n': n, 'note': note})

    result = pd.DataFrame(rows)
    if result.empty:
        print(f"No usable data for {label_a} vs {label_b} in any bucket.")
        return result

    result['age_bucket'] = pd.Categorical(result['age_bucket'], categories=list(age_buckets), ordered=True)
    result['abs_corr'] = result['correlation'].abs()
    result = result.sort_values(['age_bucket', 'abs_corr'], ascending=[True, False]).drop(columns='abs_corr')

    print(f"\nPairwise separability (age-stratified, log1p space): {label_a} vs. {label_b}")
    print(f"(positive correlation = higher feature value favors {label_b}; negative = favors {label_a})")
    print(result.to_string(index=False))

    return result


def _parse_trend_array_str(x):
    if not isinstance(x, str):
        return x
    s = x.strip('[]').split()
    return np.array([float(i) for i in s])


PAIRS = {
    "baby": [
        (('medium_popular', 'increase'), ('zero_popular', 'stable')),
        (('medium_popular', 'stable'), ('zero_popular', 'decrease')),
        (('high_popular', 'increase'), ('medium_popular', 'decrease')),
    ],
    "home_and_kitchen": [       
        (('medium_popular', 'increase'), ('zero_popular', 'stable')),
        (('medium_popular', 'stable'), ('zero_popular', 'decrease')),
        (('high_popular', 'increase'), ('medium_popular', 'decrease')),
    ],
    "douban": [
        (('medium_popular', 'increase'), ('zero_popular', 'stable')),
        (('medium_popular', 'increase'), ('medium_popular', 'stable')),
        (('medium_popular', 'increase'), ('zero_popular', 'decrease')),
        (('medium_popular', 'stable'), ('zero_popular', 'decrease')),
        (('medium_popular', 'stable'), ('zero_popular', 'stable')),
    ],
}


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

    if 'is_single_real_trend_point' not in dataset.columns:
        dataset = dataset.copy()
        dataset['is_single_real_trend_point'] = dataset['trend_array'].apply(_is_single_real_trend_point)

    results, non_debut_with_trends = build_group_characteristics(dataset, embed)
    non_debut_df = add_embedding_similarity(non_debut_with_trends, embed, results)
    debut_df = build_debut_rows(dataset, embed, results)
    combined_df = pd.concat([non_debut_df, debut_df], ignore_index=True, sort=False)

    full_results = run_all_age_stratified_studies(combined_df)
    debut_result = debut_correlation_study(combined_df)
    bucket_composition = report_bucket_composition(combined_df)

    characteristics = ['latest_pop_value', 'item_age', 'trend_min', 'trend_max',
                        'trend_volatility', 'trend_overall_change',
                        'time_since_peak', 'latest_momentum', 'embed_similarity']
    pairs = PAIRS[name]
    if not pairs:
        print("\nNo `pairs` configured yet -- see this script's module docstring "
              "(point 4): fill in `pairs` from this category's own s6/s7 "
              "correlation results before running this comparison.")
    else:
        out_path = f"{out_dir}/{args.dataset}_pairwise_separability_stratified_output.txt"
        with open(out_path, "w") as f:
            with contextlib.redirect_stdout(f):
                for group_a, group_b in pairs:
                    pairwise_group_separability_stratified(
                        combined_df, group_a, group_b, characteristics,
                        age_buckets=('debut', 'recent_single_point', 'recent_multi_point', 'maturing'),
                    )
        print(f"Saved to {out_path}")
