"""
Build the scored member index used for retrieval: for every item in each
(group, history-depth bucket) cell, score it against its own bucket's center
via composite_score (from refined_retrieval_utils.py) and sort descending.

Works for Douban and Amazon (baby / home_and_kitchen); per-dataset settings
come from dataset_config.py. Steps: per-bucket fixed feature sets, building
each item's query_values dict, building each bucket's center dict from its
cleaned stats, scoring every member against its own bucket's center, and
sorting descending. trend_array is re-parsed from its saved-CSV string form
with float() (Amazon values aren't guaranteed integer-valued).

Run per dataset (after s5_save_eight_groups.py has produced that dataset's
group_profiles.json / group_mean_embeds.npz / group_doc_ids.npz /
completed_train_data.csv):
    python s16_indexing.py --dataset baby
"""
import argparse
import json

import numpy as np
import pandas as pd

from s5_save_eight_groups import (
    calculate_normalized_momentum,
    trend_summary_stats,
    load_group_profiles,
)
from refined_retrieval_utils import composite_score
from dataset_config import DATASETS

BUCKET_ORDER = ('debut', 'recent_single_point', 'recent_multi_point', 'maturing')


# ---------------------------------------------------------------------------
# per-item feature dict (bucket/category is already known from the caller
# -- this just builds the query_values dict, no re-derivation of bucket)
# ---------------------------------------------------------------------------

def build_query_values(row, embed_vector):
    stats = trend_summary_stats(row["trend_array"])
    return {
        "latest_pop_value": float(row["latest_pop_value"]),
        "item_age": float(row["item_age"]),
        "trend_min": stats["trend_min"],
        "trend_max": stats["trend_max"],
        "trend_volatility": stats["trend_volatility"],
        "trend_overall_change": stats["trend_overall_change"],
        "time_since_peak": stats["time_since_peak"],
        "latest_momentum": row["latest_momentum"] if row["latest_momentum"] != "unavailable" else None,
        "embed_vector": embed_vector,
    }


# ---------------------------------------------------------------------------
# center dict -- built from the BUCKET's own cleaned stats (_clean_debut_
# stats / _clean_single_point_stats / _clean_full_stats output in
# s5_save_eight_groups.py), not the group's pooled stats. Field
# names must match those cleaners exactly -- note time_since_peak's real
# key is 'time_since_peak_median_multi_point', not 'time_since_peak_median'.
# ---------------------------------------------------------------------------

_FEATURE_TO_CENTER_KEY_LOCAL = {
    "latest_pop_value": "latest_pop_value_median",
    "item_age": "item_age_median",
    "trend_min": "trend_min_median",
    "trend_max": "trend_max_median",
    "trend_volatility": "trend_volatility_median",
    "trend_overall_change": "trend_overall_change_median",
    "time_since_peak": "time_since_peak_median_multi_point",
    "latest_momentum": "latest_mom_median",
    "embed_similarity": "embed_vector",
}

# fixed per-bucket feature set + equal weights -- NOT derived from
# usable_features_equal_weight's query-side "is this None" check, because
# a bucket's available fields are a property of the BUCKET TYPE (what
# _clean_*_stats even stores), not of any individual query item.
_BUCKET_FEATURES = {
    "debut": (["embed_similarity"], [1.0]),
    "recent_single_point": (["latest_pop_value", "embed_similarity"], [0.5, 0.5]),
    # recent_multi_point / maturing built dynamically below, from the full set
}

_FULL_BUCKET_FEATURE_SET = [
    "latest_pop_value", "item_age", "trend_min", "trend_max",
    "trend_volatility", "trend_overall_change", "time_since_peak",
    "latest_momentum", "embed_similarity",
]


def _build_center_dict(bucket, bucket_stats, mean_embed_vector):
    if bucket == "debut":
        return {"embed_vector": mean_embed_vector}
    if bucket == "recent_single_point":
        return {
            "latest_pop_value_median": bucket_stats["latest_pop_value_median"],
            "embed_vector": mean_embed_vector,
        }
    # recent_multi_point / maturing: full set
    return {
        "latest_pop_value_median": bucket_stats["latest_pop_value_median"],
        "item_age_median": bucket_stats["item_age_median"],
        "trend_min_median": bucket_stats["trend_min_median"],
        "trend_max_median": bucket_stats["trend_max_median"],
        "trend_volatility_median": bucket_stats["trend_volatility_median"],
        "trend_overall_change_median": bucket_stats["trend_overall_change_median"],
        "time_since_peak_median_multi_point": bucket_stats["time_since_peak_median_multi_point"],
        "latest_mom_median": bucket_stats["latest_mom_median"],
        "embed_vector": mean_embed_vector,
    }


def _feature_set_for_bucket(bucket, query_values, center):
    """Returns (usable_feats, weights) for this bucket. debut/
    recent_single_point are fixed (see _BUCKET_FEATURES) -- never
    recomputed here, so their weights can't silently dilute. recent_multi_
    point/maturing use the full set, minus any feature missing (None) or
    unmeasurable (NaN) on either the query or center side -- NaN must be
    checked explicitly, since a fallback-borrowed pooled center can have a
    NaN field (e.g. an empty peak_source median) that `is not None` alone
    would let through."""
    if bucket in _BUCKET_FEATURES:
        return _BUCKET_FEATURES[bucket]

    usable = [
        f for f in _FULL_BUCKET_FEATURE_SET
        if pd.notna(query_values.get(f))
        and pd.notna(center.get(_FEATURE_TO_CENTER_KEY_LOCAL[f]))
    ]
    if not usable:
        return ["embed_similarity"], [1.0]
    return usable, [1.0 / len(usable)] * len(usable)


# ---------------------------------------------------------------------------
# main indexing pass (unchanged)
# ---------------------------------------------------------------------------

def build_scored_member_index(dataset, embed, profiles, mean_embeds, doc_ids_by_group,
                               doc_id_col="doc_id", trend_col="trend_array"):
    """
    Returns: nested dict, group_key -> bucket -> list of (doc_id, score)
    sorted descending. E.g. scored_index["high_popular__increase"]["maturing"].
    """
    dataset = dataset.copy()
    if "latest_momentum" not in dataset.columns:
        dataset["latest_momentum"] = dataset[trend_col].apply(calculate_normalized_momentum)
    dataset_by_doc_id = dataset.set_index(doc_id_col, drop=False)

    scored_index = {}

    for group_key, profile in profiles.items():
        scored_index[group_key] = {}

        for bucket in BUCKET_ORDER:
            bucket_stats = profile.get("buckets", {}).get(bucket)
            if bucket_stats is None:
                continue  # this group has no members in this bucket

            full_key = f"{group_key}__{bucket}"
            if full_key not in doc_ids_by_group:
                continue

            mean_embed_vector = mean_embeds[full_key]
            center = _build_center_dict(bucket, bucket_stats, mean_embed_vector)

            if bucket_stats.get("is_fallback"):
                print(f"  note: {full_key} is a fallback cell -- its stats were "
                      f"borrowed from {group_key}'s pooled block, not its own real members")

            scored_members = []
            for doc_id in doc_ids_by_group[full_key]:
                doc_id = int(doc_id)
                row = dataset_by_doc_id.loc[doc_id]
                embed_vector = embed[doc_id]
                query_values = build_query_values(row, embed_vector)
                usable_feats, weights = _feature_set_for_bucket(bucket, query_values, center)
                score = composite_score(usable_feats, query_values, center, weights=weights)

                scored_members.append((doc_id, round(float(score), 4)))

            scored_members.sort(key=lambda t: t[1], reverse=True)
            scored_index[group_key][bucket] = scored_members

            print(f"{full_key}: scored {len(scored_members)} members "
                  f"(top={scored_members[0][1] if scored_members else 'n/a'}, "
                  f"bottom={scored_members[-1][1] if scored_members else 'n/a'})")

    return scored_index


def save_scored_member_index(scored_index, out_path="group_scored_members.json"):
    with open(out_path, "w") as f:
        json.dump(scored_index, f, indent=2)
    total = sum(len(bucket_list) for buckets in scored_index.values() for bucket_list in buckets.values())
    print(f"Saved scored member index ({total} total items across "
          f"{sum(len(b) for b in scored_index.values())} (group, bucket) cells) -> {out_path}")


def load_scored_member_index(path="group_scored_members.json"):
    with open(path) as f:
        return json.load(f)


def top_k(scored_index, group_key, bucket, k=10):
    """Convenience: top-k (doc_id, score) for a given (group, bucket)."""
    return scored_index.get(group_key, {}).get(bucket, [])[:k]


def _parse_trend_array_str(x):
    """Re-parses trend_array from its saved-CSV string form, e.g.
    '[1. 1. 2.]' or '[]'. Uses float() (not the movie reference's int()) --
    Amazon values aren't guaranteed integer-valued."""
    if not isinstance(x, str):
        return x
    s = x.strip('[]').split()
    return np.array([float(i) for i in s])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--completed-csv', default=None)
    parser.add_argument('--embed-npy', default=None)
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    name = args.dataset
    completed_csv = args.completed_csv or f"{cfg['out_dir']}/{name}_completed_train_data.csv"
    embed_npy = args.embed_npy or cfg["embed"]
    out_path = args.out_path or f"{cfg['out_dir']}/{name}_group_scored_members.json"

    dataset = pd.read_csv(completed_csv)
    embed = np.load(embed_npy, mmap_mode='r')
    assert len(dataset) == len(embed), (
        f"row-count mismatch: dataset has {len(dataset)} rows but "
        f"embed has {len(embed)} -- doc_id positions won't line up with embed rows"
    )
    dataset["trend_array"] = dataset["trend_array"].apply(_parse_trend_array_str)

    profiles, mean_embeds, doc_ids_by_group = load_group_profiles(cfg['out_dir'])
    scored_index = build_scored_member_index(dataset, embed, profiles, mean_embeds, doc_ids_by_group)
    save_scored_member_index(scored_index, out_path=out_path)
