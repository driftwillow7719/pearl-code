"""
convert_npz_to_long.py -- Convert the personalized models' dense-matrix NPZ export
(Baby / Douban, scores.npz + users.csv + items.csv + test_targets.csv) into
the SAME long-format files used for Kitchen, so s23_score_mixture_multidataset.py
and s24_evaluate_hr_ndcg_2.py can be reused WITHOUT any changes.

Output (per dataset), matching Kitchen's existing layout exactly:
    <out_dir>/HGN/user_item_scores.csv.gz      -- user_id, item_id, original_item_id, hgn_score
    <out_dir>/LLMESR/user_item_scores.csv.gz   -- user_id, item_id, original_item_id, llmesr_score
    <out_dir>/test_interactions.csv            -- user_id, item_id

Column meaning (per README in the new export):
    user_id         <- npz `user_ids`          (original author/reviewer ID -- what
                                                  test_targets.csv uses)
    item_id         <- npz `catalog_item_ids`  (Baby: catalog item_id, Douban: movie_id --
                                                  what test_targets.csv and the
                                                  popularity file should join on)
    original_item_id <- npz `candidates`        (internal MODEL item id -- used ONLY for
                                                  tie-breaking, exactly reproducing the
                                                  README's "ties broken by ascending model
                                                  item ID" rule, since `candidates` is
                                                  already in ascending order and
                                                  s23_score_mixture_multidataset.py's tie-break sorts
                                                  ascending on this column)

Both user_id and item_id are kept as STRINGS throughout (per README: Baby ASINs /
reviewer IDs need this to avoid losing leading zeros) -- join your popularity file
on item_id as a string too, or cast both sides consistently.

Usage:
    python convert_npz_to_long.py --dataset_dir Baby --out_dir converted/Baby
    python convert_npz_to_long.py --dataset_dir Douban --out_dir converted/Douban
"""
import argparse
import os

import numpy as np
import pandas as pd


def load_npz_long(npz_path: str, score_col_name: str) -> pd.DataFrame:
    """Unroll a dense (users x candidates) scores.npz into long-format rows,
    fully vectorized (np.repeat/np.tile), no python-level row loop."""
    with np.load(npz_path, allow_pickle=False) as data:
        scores = data["scores"]  # (n_users, n_candidates) float32
        user_ids = data["user_ids"].astype(str)  # (n_users,)
        catalog_item_ids = data["catalog_item_ids"].astype(str)  # (n_candidates,)
        candidates = data["candidates"]  # (n_candidates,) model item id, ascending

    n_users, n_candidates = scores.shape
    assert len(user_ids) == n_users, f"user_ids len {len(user_ids)} != n_users {n_users}"
    assert len(catalog_item_ids) == n_candidates, (
        f"catalog_item_ids len {len(catalog_item_ids)} != n_candidates {n_candidates}"
    )
    assert np.all(np.diff(candidates) > 0), (
        "candidates array isn't strictly ascending -- tie-break assumption from the "
        "README ('ties broken by ascending model item ID', reproduced by stable sort "
        "over this array's existing order) doesn't hold, check the export"
    )

    df = pd.DataFrame({
        "user_id": np.repeat(user_ids, n_candidates),
        "item_id": np.tile(catalog_item_ids, n_users),
        "original_item_id": np.tile(candidates, n_users),  # tie-break key only
        score_col_name: scores.reshape(-1),
    })
    print(f"[load] {npz_path}: {len(df)} rows ({n_users} users x {n_candidates} candidates), "
          f"{df['user_id'].nunique()} unique users, {df['item_id'].nunique()} unique items")
    return df


def convert_test_targets(test_targets_path: str) -> pd.DataFrame:
    """test_targets.csv has original_user_id + (item id or movie_id) --
    rename to user_id/item_id to match the converted score files and what
    s24_evaluate_hr_ndcg_2.py's --test_interactions_path expects."""
    df = pd.read_csv(test_targets_path, dtype=str)
    cols = list(df.columns)
    assert cols[0] == "original_user_id", f"unexpected first column: {cols[0]}"
    item_col = cols[1]  # the Baby item id column or "movie_id" (Douban)
    df = df.rename(columns={"original_user_id": "user_id", item_col: "item_id"})
    print(f"[targets] {test_targets_path}: {len(df)} rows, {df['user_id'].nunique()} users "
          f"(item id column was '{item_col}')")
    return df[["user_id", "item_id"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", required=True,
                    help="e.g. Baby or Douban (folder containing HGN/, LLMESR/, test_targets.csv)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    hgn_npz = os.path.join(args.dataset_dir, "HGN", "scores.npz")
    llmesr_npz = os.path.join(args.dataset_dir, "LLMESR", "scores.npz")
    test_targets_path = os.path.join(args.dataset_dir, "test_targets.csv")

    hgn_out_dir = os.path.join(args.out_dir, "HGN")
    llmesr_out_dir = os.path.join(args.out_dir, "LLMESR")
    os.makedirs(hgn_out_dir, exist_ok=True)
    os.makedirs(llmesr_out_dir, exist_ok=True)

    hgn_df = load_npz_long(hgn_npz, "hgn_score")
    hgn_out_path = os.path.join(hgn_out_dir, "user_item_scores.csv.gz")
    hgn_df.to_csv(hgn_out_path, index=False, compression="gzip")
    print(f"[write] {hgn_out_path}")
    del hgn_df

    llmesr_df = load_npz_long(llmesr_npz, "llmesr_score")
    llmesr_out_path = os.path.join(llmesr_out_dir, "user_item_scores.csv.gz")
    llmesr_df.to_csv(llmesr_out_path, index=False, compression="gzip")
    print(f"[write] {llmesr_out_path}")
    del llmesr_df

    targets_df = convert_test_targets(test_targets_path)
    targets_out_path = os.path.join(args.out_dir, "test_interactions.csv")
    targets_df.to_csv(targets_out_path, index=False)
    print(f"[write] {targets_out_path}")

    print("\nDone. Point s23_score_mixture_multidataset.py at:")
    print(f"  --hgn_scores_path {hgn_out_path}")
    print(f"  --llmesr_scores_path {llmesr_out_path}")
    print("and s24_evaluate_hr_ndcg_2.py at:")
    print(f"  --test_interactions_path {targets_out_path}")
    print("--combined_id_col item_id (unchanged -- same convention as Kitchen)")


if __name__ == "__main__":
    main()
