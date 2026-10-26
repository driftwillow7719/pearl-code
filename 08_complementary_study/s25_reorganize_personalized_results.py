"""
s25_reorganize_personalized_results.py — Reorganize pare_hr_ndcg_results_kitchen.csv:
  - keep only: variant, popularity_col, model, beta, hr@5, hr@10, reference_ndcg@5, reference_ndcg@10
  - order rows by model, then beta (increasing)
  - split into separate files per popularity_col (norm_predict_popularity_score, norm_rank10, norm_rank20)

Note: the new pipeline (the personalized models' full-candidate export) doesn't have retrieval
variants anymore (no more k1_full/k5_full/k10_full/no_retrieval distinction --
that was tied to the old candidate-retrieval pipeline).
"""
import os
import pandas as pd


import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--in-path", required=True, help="HR/NDCG results CSV produced by s24")
_p.add_argument("--out-dir", required=True, help="directory for the per-popularity-column files")
_args = _p.parse_args()
IN_PATH, OUT_DIR = _args.in_path, _args.out_dir

KEEP_COLS = ["popularity_col", "model", "beta", "hr@5", "hr@10", "reference_ndcg@5", "reference_ndcg@10"]

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(IN_PATH)
df = df[KEEP_COLS].copy()
df = df.sort_values(["model", "beta"]).reset_index(drop=True)

for col, grp in df.groupby("popularity_col"):
    grp = grp.drop(columns=["popularity_col"]).copy()
    grp.insert(0, "variant", "k10_full")  # hardcoded -- see note above
    out_path = f"{OUT_DIR}/results_{col}.csv"
    grp.to_csv(out_path, index=False)
    print(f"saved {len(grp)} rows -> {out_path}")
