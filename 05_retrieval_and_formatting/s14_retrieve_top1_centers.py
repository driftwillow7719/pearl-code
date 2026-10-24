"""
Top-1 center retrieval for Douban and Amazon (baby / home_and_kitchen).

The script has two scoring modes, --feature-selector-active /
--no-feature-selector-active, and two random baselines (mutually exclusive):
--random-group picks the top-1 center uniformly at random (no scoring), and
--random-features, described below, which isolates whether the LLM feature
selector's CHOICE of features matters, independent of its weighting scheme:

  ACTIVE   -- LLM-ranked top-2/top-3 features, rank-weighted
    per RANK_WEIGHTS (3 features -> 0.50/0.30/0.20, 2 -> 0.70/0.30).

  INACTIVE -- every usable feature for the item's bucket, equal
    weight (1/n each). No LLM call.

  RANDOM-FEATURES -- draws --n-random-features (default 3) features
    UNIFORMLY AT RANDOM from the same usable-feature set inactive mode uses
    for that item's bucket (via usable_features_equal_weight's feature
    list), then scores them with the SAME RANK_WEIGHTS scheme active mode
    uses (passing weights=None lets composite_score fall back to
    RANK_WEIGHTS[len(selected)] exactly like active mode does) -- so the
    weighting formula is held constant and only the feature IDENTITIES (and
    their rank-weight assignment, since the random draw order becomes the
    rank order) are randomized. No LLM call needed.

    For recent_single_point items, the usable-feature set is only
    {latest_pop_value, embed_similarity} (2 features -- every trend-derived
    feature needs >=2 recorded points), so n_random_features is capped at
    however many are usable (min(n_random_features, len(usable))), landing
    on RANK_WEIGHTS[2] (0.70/0.30) automatically for that bucket. If a
    bucket ever has fewer usable features than RANK_WEIGHTS defines a tier
    for, composite_score's existing KeyError handling (already present in
    retrieve_top_1's per-candidate-key loop) records that row as an error
    rather than crashing the run -- same safety net the other two modes
    already rely on.

    DEBUT items are UNCHANGED in this mode too: always embed_similarity
    alone (weight 1.0), same structurally-possible-centers restriction --
    random-features never touches debut items, exactly like active/inactive
    don't either.

Run, e.g.:
    python s14_retrieve_top1_centers.py --dataset douban \\
        --random-features --n-random-features 3 --random-seed 1
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "03_clustering_and_indexing"))
from refined_retrieval_utils import (
    load_center_profiles,
    composite_score,
    usable_features_equal_weight,
)
from dataset_config import DATASETS

MATURING_TREND_POINT_THRESHOLD = 3  # keep in sync with s5_save_eight_groups.py

# for Amazon only increases but Douban with zero popular stable
DEBUT_CANDIDATE_KEYS = [
    "zero_popular__stable__pooled",
    "medium_popular__increase__pooled",
    "high_popular__increase__pooled",
]
FEATURE_NAME_ALIASES = {
    "momentum": "latest_momentum",
    # add other known LLM naming slips here as they turn up
}
# fixed, deterministic ordering for the INACTIVE (equal-weight) baseline --
# order doesn't affect the score (all weights equal), but keeps output
# reproducible/comparable across runs
ALL_CANDIDATE_FEATURES = [
    "latest_pop_value", "item_age", "trend_min", "trend_max",
    "trend_volatility", "trend_overall_change", "time_since_peak",
    "latest_momentum", "embed_similarity",
]

_POP_HIGH = None   # set once in __main__ from the chosen dataset

# ======== POP CLUSTER THRESHOLD ==========
def popularity_threshold(pop_value):
    """Zero / medium / high. 'high' starts at DATASETS[...]['high']."""
    if pop_value == 0:
        return "zero_popular"
    elif pop_value > 0 and pop_value < _POP_HIGH:
        return "medium_popular"
    else:
        return "high_popular"

# ======== RETRIEVED DIRECTION ==========
def get_retrieved_direction(item_age, popularity_value, ground_truth_score) -> str:
    if int(item_age) == -1:
        if ground_truth_score > 0:
            return 'increase'
        else:
            return 'stable'
    else:
        if ground_truth_score > popularity_value:
            return 'increase'
        elif ground_truth_score < popularity_value:
            return 'decrease'
        else:
            return 'stable'

# ---------------------------------------------------------------------------
# feature computation (same math as s11_feature_slx_prep_batch.py)
# ---------------------------------------------------------------------------

def has_real_history(trend_array):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) == 0:
        return False
    if len(trend_array) == 1 and trend_array[0] == -1:
        return False
    return True


def trend_summary_stats(trend_array):
    arr = np.asarray(trend_array, dtype=float)
    trend_min = float(arr.min())
    trend_max = float(arr.max())
    trend_volatility = float(arr.std()) if len(arr) > 1 else None
    trend_overall_change = float(arr[-1] - arr[0]) if len(arr) > 1 else None
    peak_idx = int(arr.argmax())
    time_since_peak = (len(arr) - 1) - peak_idx
    return {
        "trend_min": trend_min, "trend_max": trend_max,
        "trend_volatility": trend_volatility,
        "trend_overall_change": trend_overall_change,
        "time_since_peak": time_since_peak,
    }


def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return None
    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w + 1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def determine_category(item_age):
    """debut / recent_single_point / recent_multi_point / maturing --
    based on item_age alone, matching the canonical get_history_depth_bucket
    used to build the training centers. NOT based on trend_array length.
    Lowercase (unlike the movie reference's uppercase DEBUT/etc) to match
    usable_features_equal_weight's expected bucket-name casing directly."""
    if item_age == -1:
        return "debut"
    elif item_age == 0:
        return "recent_single_point"
    elif item_age <= MATURING_TREND_POINT_THRESHOLD:
        return "recent_multi_point"
    else:
        return "maturing"


def compute_item_features(item_age, trend_array):
    """Returns (category, feature_values_dict). feature_values_dict uses
    None for anything undefined -- composite_score will raise clearly if
    an undefined feature was somehow selected anyway."""
    category = determine_category(item_age)

    if category == "debut":
        return category, {
            "latest_pop_value": None, "item_age": item_age, "trend_min": None,
            "trend_max": None, "trend_volatility": None, "trend_overall_change": None,
            "time_since_peak": None, "latest_momentum": None,
        }

    stats = trend_summary_stats(trend_array)
    momentum = calculate_normalized_momentum(trend_array)
    return category, {
        "latest_pop_value": float(trend_array[-1]),
        "item_age": item_age,
        "trend_min": stats["trend_min"],
        "trend_max": stats["trend_max"],
        "trend_volatility": stats["trend_volatility"],
        "trend_overall_change": stats["trend_overall_change"],
        "time_since_peak": stats["time_since_peak"],
        "latest_momentum": momentum,
    }


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _parse_trend_array_str(x):
    """Re-parses trend_array from its saved-CSV string form, e.g.
    '[1. 1. 2.]' or '[]'. Uses float() (not the movie reference's int()) --
    Amazon values aren't guaranteed to print as bare integers."""
    if not isinstance(x, str):
        return x
    s = x.strip("[]").split()
    return np.array([float(i) for i in s])


def load_query_data(query_csv, embed_path, id_col="doc_id", parse_selected_features=True):
    df = pd.read_csv(query_csv)
    df["trend_array"] = df["trend_array"].apply(_parse_trend_array_str)
    if id_col not in df.columns:
        df[id_col] = list(range(len(df)))
    if parse_selected_features and "selected_features" in df.columns:
        df["selected_features"] = df["selected_features"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["selected_features"] = df["selected_features"].apply(normalize_selected_features)  # <-- add this line
    embed = np.load(embed_path, mmap_mode="r")
    return df, embed

def attach_ground_truth_direction(df):
    """Amazon-specific addition (not present in the movie reference, which
    received these columns pre-attached by a separate, unincluded step):
    computes ground_truth_pop_cluster / ground_truth_direction from
    ground_truth_popularity_score + latest_pop_value + item_age, using the
    SAME popularity_threshold / get_retrieved_direction functions used to
    build the training-side group labels in s5_save_eight_groups.py
    -- so a query item's ground-truth group is defined identically to how
    training items were grouped, rather than left as an unfillable None."""
    if "ground_truth_popularity_score" not in df.columns:
        print("note: no 'ground_truth_popularity_score' column found -- "
              "ground_truth_pop_cluster/ground_truth_direction will not be computed.")
        return df
    df = df.copy()
    df["ground_truth_pop_cluster"] = df["ground_truth_popularity_score"].apply(popularity_threshold)
    df["ground_truth_direction"] = df.apply(
        lambda row: get_retrieved_direction(
            row["item_age"], row["latest_pop_value"], row["ground_truth_popularity_score"]
        ),
        axis=1,
    )
    return df

def normalize_selected_features(selected):
    """Fixes up feature names the LLM occasionally gets slightly wrong
    (e.g. 'momentum' instead of the canonical 'latest_momentum'), and
    dedupes in case normalization creates a repeat -- important since
    RANK_WEIGHTS is keyed by len(selected) (2 or 3), so a duplicate would
    silently pick the wrong weight tier."""
    if not isinstance(selected, list):
        return selected
    mapped = [FEATURE_NAME_ALIASES.get(f, f) for f in selected]
    seen = set()
    deduped = []
    for f in mapped:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped

# ---------------------------------------------------------------------------
# retrieval (unchanged logic from the movie reference, plus the new
# random_features branch)
# ---------------------------------------------------------------------------

def retrieve_top_1(df, embed, centers, feature_selector_active=True,
                    id_col="doc_id", gamma_pop=1.0, gamma_mom=1.0, beta_mom=1.0,
                    random_group=False, random_features=False, n_random_features=3,
                    rng=None):
    all_center_keys = list(centers.keys())
    results = []

    for _, row in df.iterrows():
        ground_truth_cols = {
            "item_id": row.get("item_id"),
            "ground_truth_direction": row.get("ground_truth_direction"),
            "ground_truth_pop_cluster": row.get("ground_truth_pop_cluster"),
            "ground_truth_popularity_score": row.get("ground_truth_popularity_score"),
        }

        if random_group:
            category = determine_category(row["item_age"])
            # same candidate constraint as the semantic mode: debut items can only
            # land in the structurally possible centers
            candidate_keys = sorted(DEBUT_CANDIDATE_KEYS if category == "debut" else all_center_keys)
            top_key = candidate_keys[int(rng.integers(len(candidate_keys)))]
            results.append({
                "custom_id": row.get("custom_id"), "doc_id": row.get(id_col),
                "category": category, "mode": "random",
                "selected_features": None, "top_1_center": top_key,
                "top_1_score": None, "error": None,
                **ground_truth_cols,
            })
            continue

        category, feats = compute_item_features(row["item_age"], row["trend_array"])
        feats["embed_vector"] = embed[int(row[id_col])]

        weights = None  # None -> composite_score falls back to RANK_WEIGHTS lookup

        if category == "debut":
            # always embed_similarity alone, regardless of mode (active /
            # inactive / random_features all hit this branch identically)
            candidate_keys = DEBUT_CANDIDATE_KEYS
            selected = ["embed_similarity"]
            weights = [1.0]

        elif random_features:
            # NEW: draw n_random_features features UNIFORMLY AT RANDOM from
            # the same usable-feature set inactive mode would use for this
            # item's bucket -- then score with weights=None so
            # composite_score falls back to the SAME RANK_WEIGHTS scheme
            # active mode uses (0.50/0.30/0.20 for 3, 0.70/0.30 for 2). Only
            # the feature IDENTITIES (and their resulting rank-weight slot,
            # since draw order becomes rank order) are randomized -- the
            # weighting formula itself is held constant, to isolate whether
            # the LLM's choice of features (vs. a random choice) matters.
            candidate_keys = all_center_keys
            usable_all, _ = usable_features_equal_weight(feats, category)
            n_pick = min(n_random_features, len(usable_all))
            if n_pick == 0:
                results.append({
                    "custom_id": row.get("custom_id"), "doc_id": row.get(id_col), "category": category,
                    "mode": "random_features", "selected_features": None, "top_1_center": None,
                    "top_1_score": None, "error": "no usable features for this item's bucket",
                    **ground_truth_cols,
                })
                continue
            chosen_idx = rng.choice(len(usable_all), size=n_pick, replace=False)
            selected = [usable_all[i] for i in chosen_idx]
            # weights stays None -> RANK_WEIGHTS[len(selected)]; if n_pick
            # ever lands outside RANK_WEIGHTS' defined tiers (2 or 3), the
            # existing KeyError handling in the scoring loop below records
            # this row as an error rather than crashing the run.

        elif feature_selector_active:
            candidate_keys = all_center_keys
            selected = row.get("selected_features")
            if not isinstance(selected, list) or not selected:
                results.append({
                    "custom_id": row.get("custom_id"), "doc_id": row.get(id_col), "category": category,
                    "mode": "active", "selected_features": None, "top_1_center": None,
                    "top_1_score": None, "error": "no valid selected_features found on this row",
                    **ground_truth_cols,
                })
                continue
            # weights stays None -> RANK_WEIGHTS[len(selected)] (0.50/0.30/0.20 or 0.70/0.30)

        else:
            # INACTIVE baseline: every usable feature, equal weight
            candidate_keys = all_center_keys
            selected, weights = usable_features_equal_weight(feats, category)

        scores = {}
        skip = False
        for key in candidate_keys:
            try:
                scores[key] = composite_score(
                    selected, feats, centers[key], gamma_pop, gamma_mom, beta_mom, weights=weights
                )
            except KeyError as e:
                mode_label = ("random_features" if (random_features and category != "debut")
                              else "active" if feature_selector_active else "inactive")
                results.append({
                    "custom_id": row.get("custom_id"), "doc_id": row.get(id_col), "category": category,
                    "mode": mode_label,
                    "selected_features": selected, "top_1_center": None, "top_1_score": None,
                    "error": str(e),
                })
                skip = True
                break
        if skip:
            continue

        top_key = max(scores, key=scores.get)
        mode_label = ("random_features" if (random_features and category != "debut")
                      else "active" if feature_selector_active else "inactive")
        record = {
            "custom_id": row.get("custom_id"),
            "doc_id": row.get(id_col),
            "category": category,
            "mode": mode_label,
            "selected_features": selected,
            "top_1_center": top_key,
            "top_1_score": scores[top_key],
            "error": None,
            **ground_truth_cols,
        }
        record.update({f"score__{k}": v for k, v in scores.items()})
        results.append(record)

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the 8 (pop_cluster, pop_direction) group profiles "
                    "for one dataset (douban, baby or home_and_kitchen) -- "
                    "with a random-features scoring mode available alongside "
                    "active/inactive (--random-features)."
    )
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--training-data', default=None, help="Override the default data file.")
    parser.add_argument('--embed-npy', default=None, help="Override the default embeddings file.")
    parser.add_argument('--query-csv', default=None,
                         help="Defaults to processed_results/{name}/"
                              "{name}_feature_selection_merged.csv -- this "
                              "must already have selected_features merged onto "
                              "it from the LLM batch results (only required in "
                              "--feature-selector-active mode) plus a "
                              "history_depth_bucket/custom_id per s11's mapping "
                              "CSV output. Not required for --random-features.")
    parser.add_argument('--embed-path', default=None,
                         help="Defaults to {name}_test_embed/final_embeddings.npy")
    parser.add_argument('--random-group', action='store_true',
                        help="Ablation: pick top_1_center uniformly at random instead of "
                            "scoring. Ignores the feature selector. Mutually exclusive "
                            "with --random-features.")
    parser.add_argument('--random-features', action='store_true',
                        help="Ablation: draw --n-random-features features uniformly at "
                             "random from this item's usable-feature set (same set "
                             "--no-feature-selector-active uses), then score with the "
                             "SAME RANK_WEIGHTS scheme --feature-selector-active uses -- "
                             "isolates whether the LLM's feature CHOICE matters, holding "
                             "the weighting formula constant. No LLM call needed. "
                             "Mutually exclusive with --random-group.")
    parser.add_argument('--n-random-features', type=int, default=3,
                        help="How many features to draw per item under --random-features "
                             "(capped at however many are usable for that item's bucket -- "
                             "e.g. recent_single_point only ever has 2 usable). Default 3.")
    parser.add_argument('--random-seed', type=int, default=42,
                        help="Seed for --random-group / --random-features (use a "
                             "different seed per replicate run).")
    parser.add_argument('--id-col', default='doc_id')
    parser.add_argument('--feature-selector-active', dest='feature_selector_active',
                         action='store_true', default=True,
                         help="Use LLM-ranked selected_features (rank-weighted). Default. "
                              "Ignored if --random-features is set.")
    parser.add_argument('--no-feature-selector-active', dest='feature_selector_active',
                         action='store_false',
                         help="Equal-weight baseline over all usable features -- no LLM lookup needed.")
    parser.add_argument('--gamma-pop', type=float, default=1.0)
    parser.add_argument('--gamma-mom', type=float, default=1.0)
    parser.add_argument('--beta-mom', type=float, default=1.0)
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to processed_results/{name}")
    parser.add_argument('--save-tag', default=None,
                         help="Defaults to {name}_top_1_centers -- mode suffix "
                              "(_active/_inactive/_random_features_seedN) auto-appended, "
                              "matching the movie reference's SAVE_PATH convention.")
    args = parser.parse_args()

    if args.random_group and args.random_features:
        parser.error("--random-group and --random-features are mutually exclusive "
                      "ablations -- pick one.")

    name = args.dataset
    cfg = DATASETS[args.dataset]
    _POP_HIGH = cfg["high"]
    print("pop high threshold: ",_POP_HIGH)
    data_path = args.training_data or cfg["data"]
    embed_npy = args.embed_npy or cfg["embed"]
    print(f"Dataset: {args.dataset} | cutoffs: zero=0, medium=1..{_POP_HIGH - 1}, high>={_POP_HIGH}")

    output_dir = args.output_dir or f"processed_results/{name}"
    query_csv = args.query_csv or f"{output_dir}/{name}_feature_selection_merged.csv"
    embed_path = args.embed_path or f"{name}_test_embed/final_embeddings.npy"
    group_profiles_dir = cfg["out_dir"]
    save_tag = args.save_tag or f"{name}_top_1_centers"

    rng = np.random.default_rng(args.random_seed) if (args.random_group or args.random_features) else None

    df, embed = load_query_data(
        query_csv, embed_path, id_col=args.id_col,
        parse_selected_features=args.feature_selector_active and not args.random_group and not args.random_features,
    )
    df = attach_ground_truth_direction(df)

    centers_all = load_center_profiles(
        f"{group_profiles_dir}/group_profiles.json",
        f"{group_profiles_dir}/group_mean_embeds.npz",
    )
    pooled_centers = {k: v for k, v in centers_all.items() if k.endswith("__pooled")}
    print(f"Loaded {len(pooled_centers)} centers: {list(pooled_centers.keys())}")

    if args.random_group:
        print(f"Mode: RANDOM GROUP (seed={args.random_seed})")
    elif args.random_features:
        print(f"Mode: RANDOM FEATURES (n={args.n_random_features}, RANK_WEIGHTS scheme, seed={args.random_seed})")
    else:
        print(f"Mode: {'FEATURE_SELECTOR ACTIVE (LLM-ranked features)' if args.feature_selector_active else 'FEATURE_SELECTOR INACTIVE (equal-weight baseline)'}")

    if (args.feature_selector_active and not args.random_group and not args.random_features
            and "selected_features" not in df.columns):
        raise ValueError(
            "--feature-selector-active but the dataset has no 'selected_features' "
            "column -- merge s11's LLM batch results onto the query CSV first, or "
            "pass --no-feature-selector-active / --random-features to run a "
            "baseline that doesn't need it."
        )
    if args.random_group:
        mode_suffix = f"random_seed{args.random_seed}"
    elif args.random_features:
        mode_suffix = f"random_features{args.n_random_features}_seed{args.random_seed}"
    else:
        mode_suffix = "active" if args.feature_selector_active else "inactive"

    top1_df = retrieve_top_1(
        df, embed, pooled_centers, feature_selector_active=args.feature_selector_active,
        id_col=args.id_col, gamma_pop=args.gamma_pop, gamma_mom=args.gamma_mom, beta_mom=args.beta_mom,
        random_group=args.random_group, random_features=args.random_features,
        n_random_features=args.n_random_features, rng=rng,
    )

    n_errors = top1_df["error"].notna().sum()
    if n_errors:
        print(f"note: {n_errors} items could not be scored -- see the 'error' column")

    print(top1_df["category"].value_counts())
    print(top1_df["top_1_center"].value_counts())

    save_path = f"{output_dir}/{save_tag}_{mode_suffix}.csv"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    top1_df.to_csv(save_path, index=False)
    print(f"Saved -> {save_path}")