"""
s24_evaluate_hr_ndcg_2.py — Evaluate PARE combined-score files against ground-truth
interaction datasets, using the metrics() logic from
metrics.py in this folder
(HR, NDCG, Reference NDCG, Recall, Precision).

Two ground-truth modes:

1. --test_interactions_path (RECOMMENDED for home_and_kitchen):
   Ground truth already in the SAME user_id/item_id space as the combined
   score files (e.g. SASRec/test_interactions.csv from the personalized
   model's export). No old-ID translation needed, so no users get
   excluded for ambiguous mapping. Validated against the personalized
   model's own reported results (hr@k / ndcg@k match to float precision)
   on the full 6,437-user cohort.

2. Legacy multi-category mode (--category + --interaction_path +
   --user_mapping_path): ground truth in the OLD raw ID space, needs
   translation via a user_id -> original_test_user_id mapping file. Kept
   for douban / baby, or any home_and_kitchen
   run that still needs to go through the old mapping for some reason.
   Under this mode, users with no valid original_test_user_id (ambiguous
   identical test profiles) are excluded -- see user_mapping_path's
   original_id_status column.

Results are written incrementally (appended per file as each finishes).

Usage (direct mode, home_and_kitchen):
    python s24_evaluate_hr_ndcg_2.py \
        --test_interactions_path SASRec/test_interactions.csv \
        --combined_glob "deepseek_flash_output/amazon_home_and_kitchen/combined_kitchen_*.csv.gz" \
        --combined_id_col item_id \
        --out_path deepseek_flash_output/amazon_home_and_kitchen/pare_hr_ndcg_results_kitchen.csv

Usage (legacy mode, other categories):
    python s24_evaluate_hr_ndcg_2.py --category baby \
        --interaction_path path/to/baby_test.csv \
        --combined_glob "deepseek_flash_output/amazon_baby/combined_baby_*.csv.gz" \
        --combined_id_col original_item_id \
        --user_mapping_path path/to/user_mapping.csv \
        --out_path deepseek_flash_output/amazon_baby/pare_hr_ndcg_results_baby.csv
"""
import argparse
import glob
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from metrics import metrics


# ── direct ground truth: already in current user_id/item_id space ───────

def load_targets_direct(test_interactions_path: str) -> dict:
    """Ground truth already in the combined files' current user_id/item_id
    space (test_interactions.csv). No old-ID mapping needed, no
    users excluded -- every user in the score export has a target set here.
    """
    df = pd.read_csv(test_interactions_path, dtype={"user_id": str, "item_id": str})
    print(f"[direct ground truth] loaded {len(df)} rows, {df['user_id'].nunique()} users, "
          f"from {test_interactions_path}")
    return df.groupby("user_id")["item_id"].apply(set).to_dict()


# ── legacy ground-truth loading, per category ────────────────────────────

def load_ground_truth_raw(category: str, interaction_path: str) -> pd.DataFrame:
    if category == "home_and_kitchen":
        df = pd.read_csv(
            interaction_path, sep=' ', header=None,
            names=['user', 'item', 'rating', 'time'],
        )
        print(f"[home_and_kitchen] loaded {len(df)} rows, {df['user'].nunique()} users (raw .txt)")
        return df

    if category == "baby":
        df = pd.read_csv(interaction_path)
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("user", "user_id", "userid"):
                rename_map[c] = "user"
            elif lc in ("item", "item_id", "itemid"):
                rename_map[c] = "item"
            elif lc in ("rating", "rate"):
                rename_map[c] = "rating"
        df = df.rename(columns=rename_map)
        print(f"[{category}] loaded {len(df)} rows, {df['user'].nunique()} users (csv)")
        return df

    if category == "douban":
        df = pd.read_csv(interaction_path)
        df = df.rename(columns={"UserID": "user", "MovieID": "item", "Rate": "rating", "Time": "time"})
        df["time"] = df["time"].astype(int)
        print(f"[douban] loaded {len(df)} rows, {df['user'].nunique()} users (csv, pre time-filter)")
        return df

    raise ValueError(f"Unknown category: {category}")


def build_targets(category: str, gt_df: pd.DataFrame, douban_start=None, douban_end=None,
                   douban_tz="Asia/Shanghai") -> dict:
    if category == "douban":
        if not (douban_start and douban_end):
            raise ValueError("douban requires --douban-start and --douban-end")
        tz = ZoneInfo(douban_tz)
        start_unix = datetime.strptime(douban_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        end_unix = datetime.strptime(douban_end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        gt_df = gt_df[(gt_df["time"] >= start_unix) & (gt_df["time"] < end_unix)]
        print(f"[douban] {len(gt_df)} rows fall inside the evaluation window, "
              f"{gt_df['user'].nunique()} users")

    targets = {}
    for user_id, grp in gt_df.groupby("user"):
        targets[user_id] = set(grp["item"].tolist())
    return targets


# ── legacy user_id mapping (combined's internal user_id -> raw ground-truth user id) ──

def load_user_mapping(mapping_path: str) -> dict:
    """Returns dict: combined's user_id -> original_test_user_id (raw ground-truth
    user id). Rows with no valid original_test_user_id are dropped -- those users
    simply can't be evaluated against ground truth under this (legacy) mode."""
    mapping = pd.read_csv(mapping_path)
    before = len(mapping)
    mapping = mapping.dropna(subset=["original_test_user_id"])
    dropped = before - len(mapping)
    if dropped:
        print(f"[user_mapping] {dropped}/{before} rows have no original_test_user_id -- "
              f"those users will be excluded from evaluation (legacy mode only; use "
              f"--test_interactions_path to avoid this entirely)")
    mapping["original_test_user_id"] = mapping["original_test_user_id"].astype(int)
    user_map = dict(zip(mapping["user_id"], mapping["original_test_user_id"]))
    print(f"[user_mapping] loaded {len(user_map)} valid user_id -> original_test_user_id pairs")
    return user_map


# ── rankings from PARE combined-score files ──────────────────────────────

def build_rankings(combined: pd.DataFrame, id_col: str, user_map: dict | None) -> dict:
    """combined: 1 (model, beta) slice, already sorted by rank_in_user (which
    s23_score_mixture_multidataset.py builds with ties broken by ascending
    original_item_id, matching the model's tie_policy).

    Returns dict: target-space user_id -> ordered list of item ids
    (highest s_new first). If user_map is None (direct mode), the
    combined file's own user_id IS the target-space user_id already.
    """
    rankings = {}
    for user_id, grp in combined.sort_values("rank_in_user").groupby("user_id"):
        if user_map is not None:
            target_user_id = user_map.get(user_id)
            if target_user_id is None:
                continue  # legacy mode: no valid mapping for this user, skip
        else:
            target_user_id = user_id
        rankings[target_user_id] = grp[id_col].tolist()
    return rankings


# ── evaluate one combined-score file ─────────────────────────────────────

def evaluate_one_file(path: str, targets: dict, cutoffs: list, id_col: str, user_map: dict | None):
    combined = pd.read_csv(path, compression="gzip", dtype={"user_id": str, "item_id": str})
    fname = path.split("/")[-1]
    m_variant = re.search(r"combined_\w+?_(.+?)_pare", fname)
    m_col = re.search(r"pare_(.+?)_beta", fname)
    variant = m_variant.group(1) if m_variant else "unknown"
    popularity_col = m_col.group(1) if m_col else "unknown"

    results = []
    for (model, beta), grp in combined.groupby(["model", "beta"]):
        rankings_by_user = build_rankings(grp, id_col, user_map)

        common_users = set(targets.keys()) & set(rankings_by_user.keys())
        n_missing = len(targets) - len(common_users)

        targets_list, rankings_list, n_too_short = [], [], 0
        for uid in common_users:
            r = rankings_by_user[uid]
            if len(r) < max(cutoffs) or len(set(r)) != len(r):
                n_too_short += 1
                continue
            targets_list.append(targets[uid])
            rankings_list.append(r)

        if not targets_list:
            print(f"[warn] {variant}/{popularity_col}/{model}/beta={beta}: no evaluable users, skipping")
            continue

        res = metrics(targets_list, rankings_list, cutoffs)
        res.pop("hit_users", None)
        res.update({
            "variant": variant, "popularity_col": popularity_col,
            "model": model, "beta": beta,
            "n_users_evaluated": len(targets_list),
            "n_users_missing_from_candidates": n_missing,
            "n_users_too_few_candidates": n_too_short,
            "source_file": fname,
        })
        results.append(res)
    return results


# ── main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PARE combined-score files vs ground truth.")

    # direct mode (preferred for home_and_kitchen)
    p.add_argument("--test_interactions_path", default=None,
                    help="Ground truth CSV (user_id,item_id,...) already in the combined files' "
                         "current ID space, e.g. SASRec/test_interactions.csv. When set, this bypasses "
                         "--category/--interaction_path/--user_mapping_path entirely and no users "
                         "are excluded for ambiguous ID mapping.")

    # legacy multi-category mode
    p.add_argument("--category",
                    choices=["douban", "baby", "home_and_kitchen"],
                    help="Legacy mode only. Required unless --test_interactions_path is set.")
    p.add_argument("--interaction_path",
                    help="Legacy mode: ground-truth interaction file (.txt headerless for kitchen, "
                         ".csv for others).")
    p.add_argument("--user_mapping_path",
                    help="Legacy mode: CSV with columns user_id, original_test_user_id, "
                         "original_id_status, etc.")

    p.add_argument("--combined_glob", required=True)
    p.add_argument("--combined_id_col", default="item_id",
                    choices=["item_id", "original_item_id"],
                    help="Which id column in your combined files matches the ground truth's item "
                         "ids. 'item_id' (current ID scheme) for --test_interactions_path mode; "
                         "'original_item_id' for legacy mode with raw ground truth.")
    p.add_argument("--ks", default="5,10,20")
    p.add_argument("--out_path", default="hr_ndcg_results.csv")
    p.add_argument("--douban-start", default=None)
    p.add_argument("--douban-end", default=None)
    p.add_argument("--douban-tz", default="Asia/Shanghai")
    args = p.parse_args()

    if not args.test_interactions_path and not args.category:
        p.error("pass either --test_interactions_path (direct mode) or --category + "
                 "--interaction_path + --user_mapping_path (legacy mode)")
    return args


if __name__ == "__main__":
    args = parse_args()
    cutoffs = [int(k) for k in args.ks.split(",")]

    if args.test_interactions_path:
        targets = load_targets_direct(args.test_interactions_path)
        user_map = None  # combined's user_id is already the target-space user_id
        print(f"built targets for {len(targets)} users (direct mode -- no mapping, no exclusions)")
    else:
        if not args.interaction_path or not args.user_mapping_path:
            raise SystemExit("legacy mode needs --interaction_path and --user_mapping_path")
        gt_raw = load_ground_truth_raw(args.category, args.interaction_path)
        targets = build_targets(args.category, gt_raw, args.douban_start, args.douban_end, args.douban_tz)
        print(f"built targets for {len(targets)} users (legacy mode)")
        user_map = load_user_mapping(args.user_mapping_path)

    files = sorted(glob.glob(args.combined_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.combined_glob}")
    print(f"found {len(files)} combined-score files")

    if os.path.exists(args.out_path):
        os.remove(args.out_path)
        print(f"removed stale {args.out_path} before starting fresh append run")

    header_written = False
    total_rows = 0
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] evaluating {path} ...")
        file_results = evaluate_one_file(path, targets, cutoffs, args.combined_id_col, user_map)
        if not file_results:
            continue
        df_chunk = pd.DataFrame(file_results)
        df_chunk.to_csv(args.out_path, mode="a", header=not header_written, index=False)
        header_written = True
        total_rows += len(df_chunk)
        print(f"  -> appended {len(df_chunk)} rows (total so far: {total_rows}) -> {args.out_path}")

    print(f"\nall done. {total_rows} total result rows saved incrementally to: {args.out_path}")
