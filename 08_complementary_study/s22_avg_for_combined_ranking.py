import pandas as pd

import argparse

_p = argparse.ArgumentParser(description="Average predicted popularity over 3 runs, min-max normalize, and rank.")
_p.add_argument("--dataset", default="baby", help="dataset name used in the file names")
_p.add_argument("--base-dir", default="deepseek_flash_output/amazon_baby",
                help="directory holding run_1/ run_2/ run_3/ with <dataset>_<variant>_merged.csv")
_p.add_argument("--variants", default="k1_full,k5_full,k10_full,no_retrieval")
_p.add_argument("--runs", default="run_1,run_2,run_3")
_args = _p.parse_args()

DATASET = _args.dataset
BASE_DIR = _args.base_dir
VARIANTS = _args.variants.split(",")
RUNS = _args.runs.split(",")
ID_COL = "item_id"                      # change if your id column has a different name
SCORE_COL = "predict_popularity_score"


def load_run(run, variant):
    path = f"{BASE_DIR}/{run}/{DATASET}_{variant}_merged.csv"
    # path = f"{BASE_DIR}/{run}/{variant}_merged.csv"

    df = pd.read_csv(path)
    # scores are stored as strings -- coerce to numeric; values that failed to be
    # extracted (None/empty/odd text) become NaN instead of being counted as 0
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    return df[[ID_COL, SCORE_COL]].rename(columns={SCORE_COL: f"{SCORE_COL}_{run}"})

def add_topn_rank_norm(df, n, rank_col="rank"):
    out_col = f"norm_rank{n}"
    df[out_col] = df[rank_col].apply(
        lambda r: (n - r) / (n - 1) if r <= n else 0.0   # item outside top-n: 0, not NaN
    )
    return df

for variant in VARIANTS:
    run_dfs = [load_run(run, variant) for run in RUNS]

    merged = run_dfs[0]
    for df in run_dfs[1:]:
        merged = merged.merge(df, on=ID_COL, how="outer")

    score_cols = [c for c in merged.columns if c != ID_COL]
    # skipna=True -- if 1-2 runs failed to extract a score for an item, the
    # average is still computed from the successful runs, not treated as 0
    merged["avg_predict_popularity_score"] = merged[score_cols].mean(axis=1, skipna=True)
    # normalized version
    merged["norm_predict_popularity_score"] = (merged["avg_predict_popularity_score"] - merged["avg_predict_popularity_score"].min())/(merged["avg_predict_popularity_score"].max()-merged["avg_predict_popularity_score"].min())

    merged["n_runs_with_score"] = merged[score_cols].notna().sum(axis=1)

    ranked = merged.sort_values("avg_predict_popularity_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    
    ranked = add_topn_rank_norm(ranked,10)
    ranked = add_topn_rank_norm(ranked,20)


    out_path = f"{BASE_DIR}/{DATASET}_{variant}_final_ranking.csv"
    ranked.to_csv(out_path, index=False)
    print(f"[{variant}] -> {out_path} | {len(ranked)} items | "
          f"{ranked['avg_predict_popularity_score'].isna().sum()} items with no score in any of the 3 runs")
    a = merged["avg_predict_popularity_score"]
    print(f"actual range of predict popularity score is:{a.min()} to {a.max()}")
    n = merged["norm_predict_popularity_score"]
    print(f"range of normalized predict popularity score is:{n.min()} to {n.max()}")