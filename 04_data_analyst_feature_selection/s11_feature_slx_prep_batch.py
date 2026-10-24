"""
Amazon (baby / home_and_kitchen) and Douban.
Run per category, e.g.:
    python s11_feature_slx_prep_batch.py --category baby \\
        --prompt-path baby_feature_slx_prompt.txt
    python s11_feature_slx_prep_batch.py --category home_and_kitchen \\
        --prompt-path home_and_kitchen_feature_slx_prompt.txt
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

MATURING_TREND_POINT_THRESHOLD = 3
# change when douban stable is included
DEBUT_CANDIDATE_KEYS = ["zero_popular__stable", "medium_popular__increase", "high_popular__increase"]
# DEBUT_CANDIDATE_KEYS = ["medium_popular__increase", "high_popular__increase"]

# ---------------------------------------------------------------------------
# group profiles / embeddings
# ---------------------------------------------------------------------------

def load_group_profiles(out_dir="."):
    with open(f"{out_dir}/group_profiles.json") as f:
        profiles = json.load(f)
    mean_embeds = dict(np.load(f"{out_dir}/group_mean_embeds.npz"))
    doc_ids_by_group = dict(np.load(f"{out_dir}/group_doc_ids.npz"))
    return profiles, mean_embeds, doc_ids_by_group


def embed_scoring(embed, center_vector):
    """Cosine similarity between item embedding and group centroid, scaled to [0, 1]."""
    distance = cosine(embed, center_vector)
    similarity = 1 - distance
    return (1 + similarity) / 2

def compute_embed_similarity_dict(query_embed, mean_embeds):
    """mean_embeds: dict of "{group}__{bucket}" -> np.ndarray, loaded from
    group_mean_embeds.npz. Stage 1 group selection compares a query against
    each of the 8 groups' POOLED center only, regardless of the query's own
    history_depth_bucket -- so this filters down to just the "__pooled"
    entries and strips that suffix, returning one similarity per group (8
    entries), not per (group, bucket)."""
    pooled_embeds = {
        key[: -len("__pooled")]: vec
        for key, vec in mean_embeds.items()
        if key.endswith("__pooled")
    }
    if not pooled_embeds:
        raise ValueError(
            "No '__pooled' entries found in mean_embeds -- make sure "
            "group_mean_embeds.npz was built by save_group_profiles "
            "(with the pooled-center fix), not an older per-bucket-only version."
        )
    return {key: embed_scoring(query_embed, vec) for key, vec in pooled_embeds.items()}


# ---------------------------------------------------------------------------
# trend feature computation (same math as the training-side analysis scripts)
# ---------------------------------------------------------------------------

def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return None
    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w + 1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def trend_summary_stats(trend_array):
    arr = np.asarray(trend_array, dtype=float) if isinstance(trend_array, (list, np.ndarray)) else None
    if arr is None or len(arr) == 0 or (len(arr) == 1 and arr[0] == -1):
        return {
            "trend_min": None, "trend_max": None, "trend_volatility": None,
            "trend_overall_change": None, "time_since_peak": None,
        }
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


def has_real_history(trend_array):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) == 0:
        return False
    if len(trend_array) == 1 and trend_array[0] == -1:
        return False
    return True

def filter_debut_rows(sample_df):
    """Debut items (item_age == -1, no real trend history) never go to the
    LLM -- embed_similarity is the only usable feature for them, so their
    top-1 group is just an argmax over compute_embed_similarity_dict's
    output (nearest pooled center), not something the LLM needs to select
    among. Splits sample_df into (non_debut_df, debut_df); only
    non_debut_df gets sent to the LLM. debut_df is returned so callers can
    still fold it back into the mapping CSV with no custom_id, and route
    it through the direct nearest-pooled-center lookup instead."""
    is_debut = sample_df["item_age"] == -1
    debut_df = sample_df[is_debut].copy()
    non_debut_df = sample_df[~is_debut].copy()
    if len(debut_df):
        print(f"Filtered out {len(debut_df)} debut item(s) (item_age == -1) -- "
              f"these are resolved directly via nearest pooled embed_similarity, "
              f"never sent to the LLM.")
    return non_debut_df, debut_df

def resolve_debut_group(query_embed, mean_embeds):
    """For a debut item: skip the LLM entirely, pick the group whose
    POOLED center has the highest cosine similarity to this item's
    embedding -- restricted to the 3 outcomes a debut item can actually
    reach (zero_popular__stable, medium_popular__increase,
    high_popular__increase). The other 5 groups are logically or
    empirically unreachable for an item with no trend history, so they
    must be excluded before taking the max, not just left to lose on
    similarity score."""
    embed_sim_dict = compute_embed_similarity_dict(query_embed, mean_embeds)

    missing = set(DEBUT_CANDIDATE_KEYS) - embed_sim_dict.keys()
    if missing:
        raise ValueError(f"DEBUT_CANDIDATE_KEYS not found in pooled embeds: {missing}")

    candidate_sims = {k: embed_sim_dict[k] for k in DEBUT_CANDIDATE_KEYS}
    top_group = max(candidate_sims, key=candidate_sims.get)
    return top_group, candidate_sims

def determine_history_depth_bucket(item_age):
    """debut / recent_single_point / recent_multi_point / maturing --
    based on item_age alone (calendar months since release, as of t-1),
    matching the canonical get_history_depth_bucket used to build the
    training centers."""
    if item_age == -1:
        return "debut"
    elif item_age == 0:
        return "recent_single_point"
    elif item_age <= MATURING_TREND_POINT_THRESHOLD:
        return "recent_multi_point"
    else:
        return "maturing"

# ---------------------------------------------------------------------------
# prompt loading -- pull the static SYSTEM PROMPT block out of the
# finalized prompt file (same parsing approach as the single-call script).
# Each category's own prompt file must use the same
# "SYSTEM PROMPT (static -- identical on every call)" / "USER TURN TEMPLATE"
# section markers for this to find the block.
# ---------------------------------------------------------------------------

def load_system_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(
        r"SYSTEM PROMPT \(static -- identical on every call\)\n=+\n(.*?)\n=+\nUSER TURN TEMPLATE",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(
            "Could not find the SYSTEM PROMPT block -- check that the "
            "'====' section markers in the prompt file are still intact."
        )
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# per-row user turn construction
# ---------------------------------------------------------------------------

def build_query_block(row, bucket, embed_sim_dict):
    """row must already have passed has_real_history() -- debut rows are
    filtered out before this is ever called (see filter_debut_rows)."""
    stats = trend_summary_stats(row["trend_array"])
    momentum = calculate_normalized_momentum(row["trend_array"])
    trend_array_str = "[" + ", ".join(str(int(v)) for v in row["trend_array"]) + "]"
    fields = {
        "latest_pop_value": float(row["trend_array"][-1]),
        "item_age": row["item_age"],
        "trend_array": trend_array_str,
        "trend_min": stats["trend_min"],
        "trend_max": stats["trend_max"],
        "trend_volatility": stats["trend_volatility"] if stats["trend_volatility"] is not None else "unknown -- fewer than 2 points",
        "trend_overall_change": stats["trend_overall_change"] if stats["trend_overall_change"] is not None else "unknown -- fewer than 2 points",
        "time_since_peak": stats["time_since_peak"],
        "latest_momentum": momentum if momentum is not None else "unknown -- fewer than 2 points",
    }

    embed_sim_str = "\n".join(f"  {k}: {v:.4f}" for k, v in embed_sim_dict.items())

    return f"""history_bucket: {bucket}
latest_pop_value: {fields['latest_pop_value']}
item_age: {fields['item_age']}
trend_array (oldest to most recent, up to t-1): {fields['trend_array']}
trend_min: {fields['trend_min']}
trend_max: {fields['trend_max']}
trend_volatility: {fields['trend_volatility']}
trend_overall_change: {fields['trend_overall_change']}
time_since_peak: {fields['time_since_peak']}
latest_momentum: {fields['latest_momentum']}
embed_similarity_to_each_group (cosine similarity, precomputed -- do not
attempt to recompute from raw embeddings):
{embed_sim_str}

Select the features for this item per the rules and output format
defined in the system prompt above."""


# ---------------------------------------------------------------------------
# batch construction
# ---------------------------------------------------------------------------
def prepare_batch_input(sample_df, embed, mean_embeds, prompt_path,
                         id_col="doc_id", model="gpt-5.6-luna",
                         output_dir=".", batch_tag="feature_selection"):
    system_prompt = load_system_prompt(prompt_path)
    print(f"System prompt built ({len(system_prompt)} chars, shared across all requests).")

    non_debut_df, debut_df = filter_debut_rows(sample_df)
    print(f"Non-debut query items (sent to LLM): {len(non_debut_df)}")

    batch_requests = []
    custom_ids = []
    buckets = []
    embed_sim_records = []

    for idx, row in non_debut_df.iterrows():
        custom_id = f"{batch_tag}_request-{idx + 1:05d}"
        custom_ids.append(custom_id)

        bucket = determine_history_depth_bucket(row["item_age"])
        buckets.append(bucket)

        query_embed = embed[int(row[id_col])]
        embed_sim_dict = compute_embed_similarity_dict(query_embed, mean_embeds)
        embed_sim_records.append(embed_sim_dict)

        user_content = build_query_block(row, bucket, embed_sim_dict)

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_completion_tokens": 400,
                "temperature": 0.0,
                "reasoning_effort": "none",
                "stream": False,
            },
        }
        batch_requests.append(request)

    # mapping CSV: ALL rows (debut + non-debut), so downstream scoring still
    # covers every item -- debut rows just get no custom_id/bucket/embed_sim
    # from the LLM path (resolved directly instead).
    non_debut_df = non_debut_df.copy()
    non_debut_df["custom_id"] = custom_ids
    non_debut_df["history_depth_bucket"] = buckets
    embed_sim_df = pd.DataFrame(embed_sim_records).add_prefix("embed_sim__")
    non_debut_df = pd.concat(
        [non_debut_df.reset_index(drop=True), embed_sim_df.reset_index(drop=True)], axis=1
    )

    if len(debut_df):
        debut_df = debut_df.copy()
        debut_df["custom_id"] = None
        debut_df["history_depth_bucket"] = "debut"

        debut_top_groups = []
        debut_embed_sim_records = []
        for _, row in debut_df.iterrows():
            query_embed = embed[int(row[id_col])]
            top_group, embed_sim_dict = resolve_debut_group(query_embed, mean_embeds)
            debut_top_groups.append(top_group)
            debut_embed_sim_records.append(embed_sim_dict)

        debut_df["resolved_top_group"] = debut_top_groups
        debut_embed_sim_df = pd.DataFrame(debut_embed_sim_records).add_prefix("embed_sim__")
        debut_df = pd.concat(
            [debut_df.reset_index(drop=True), debut_embed_sim_df.reset_index(drop=True)], axis=1
        )

    mapping_df = pd.concat([non_debut_df, debut_df], ignore_index=True, sort=False)

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = f"{output_dir}/{batch_tag}_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping ({len(mapping_df)} rows, {len(debut_df)} debut) -> {mapping_path}")
    print("history_depth_bucket breakdown (mapping CSV, all rows):")
    print(mapping_df["history_depth_bucket"].value_counts().to_string())
    print(f"Created JSONL batch with {len(batch_requests)} requests (debut rows excluded)")

    return batch_requests


def save_jsonl(requests, output_dir=".", batch_tag="feature_selection"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{batch_tag}_batch.jsonl"
    with open(output_file, "w") as f:
        for item in requests:
            f.write(json.dumps(item) + "\n")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(requests)} requests -> {output_file} ({size_mb:.1f} MB)")
    return output_file


def _parse_trend_array_str(x):
    """Re-parses trend_array from its saved-CSV string form, e.g.
    '[1. 1. 2.]' or '[]'. Uses float() (not the movie reference's int()) --
    Amazon values from s5_preprocess_datasets_full.py aren't guaranteed to
    print as bare integers."""
    if not isinstance(x, str):
        return x
    s = x.strip("[]").split()
    return np.array([int(i) for i in s])


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the feature-selector LLM batch input for one "
                     "Amazon category (baby or home_and_kitchen). Each category needs "
                     "its own prompt file and group_profiles directory."
    )
    parser.add_argument('--category', choices=['baby', 'home_and_kitchen','douban'], required=True)
    parser.add_argument('--sample-csv', default=None,
                         help="Defaults to {category}_test_features.csv "
                              "(the output of s5_preprocess_datasets_full.py).")
    parser.add_argument('--embed-path', default=None,
                         help="Defaults to {category}_test_embed/final_embeddings.npy "
                              "-- must be row-aligned with --sample-csv (embed[i] is "
                              "row i's item_description embedding).")
    parser.add_argument('--group-profiles-dir', default=None,
                         help="Defaults to {category}_group_profiles (built by "
                              "s5_save_eight_groups.py).")
    parser.add_argument('--prompt-path', required=True,
                         help="This category's own feature-selection prompt file "
                              "(each category has its own).")
    parser.add_argument('--id-col', default='doc_id')
    parser.add_argument('--model', default='gpt-5.6-luna')
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to processed_results/{category}")
    parser.add_argument('--batch-tag', default=None,
                         help="Defaults to {category}_feature_selection")
    args = parser.parse_args()

    category = args.category
    sample_csv = args.sample_csv or f"{category}_test_features.csv"
    embed_path = args.embed_path or f"{category}_test_embed/final_embeddings.npy"
    group_profiles_dir = args.group_profiles_dir or f"{category}_group_profiles"
    output_dir = args.output_dir or f"processed_results/{category}"
    batch_tag = args.batch_tag or f"{category}_feature_selection"

    sample_df = pd.read_csv(sample_csv)
    sample_df["trend_array"] = sample_df["trend_array"].apply(_parse_trend_array_str)
    sample_df[args.id_col] = list(range(len(sample_df)))
    print(sample_df.info())

    embed = np.load(embed_path, mmap_mode="r")
    assert len(sample_df) == len(embed), (
        f"row-count mismatch: {sample_csv} has {len(sample_df)} rows but "
        f"{embed_path} has {len(embed)} -- {args.id_col} positions won't line up "
        f"with embed rows. Embeddings must be generated from this exact CSV, in "
        f"this exact row order."
    )

    _, mean_embeds, _ = load_group_profiles(group_profiles_dir)

    batch_requests = prepare_batch_input(
        sample_df, embed, mean_embeds, args.prompt_path,
        id_col=args.id_col, model=args.model, output_dir=output_dir, batch_tag=batch_tag,
    )
    save_jsonl(batch_requests, output_dir=output_dir, batch_tag=batch_tag)


if __name__ == "__main__":
    main()
