"""
PARE-style method integration: combine personalized user-item preference
scores (SASRec) with a non-personalized popularity score, following
Eq. 12:

    s_new(u, i, T) = beta * s(u, i) + (1 - beta) * y_hat_F

Score source (export of the personalized model):
    SASRec/user_item_scores.csv.gz  -- columns: user_id, item_id, original_item_id, sasrec_score

Normalization: sasrec_score is min-max normalized GLOBALLY
(across the whole score set), not per user -- preserves the original
relative scale of scores across users.

Popularity join: by default this is a LEFT join (not inner) -- items
with no popularity score get y_hat_F filled with --fill_value (0.0 by
default) instead of being dropped. This matters once beta < 1: dropping
those rows used to shrink the candidate pool before beta was even
applied, which silently diverged every result (even beta=1) from a
full-candidate-pool baseline. Pass --drop_missing_popularity to restore
the old inner-join + dropna behavior if you specifically need it.

Tie-breaking: within a user, ties in s_new are broken by ascending
original_item_id (tie_policy="ascending_original_item_id"), which reproduces
the personalized model's own reported baseline numbers.

Usage:
    python s23_score_mixture_multidataset.py --popularity_col norm_predict_popularity_score
"""
import argparse
import gc
import pandas as pd
import numpy as np

# ---------------- DEFAULT CONFIG (overridable via CLI) ----------------
SASREC_SCORES_PATH = "deepseek_flash_output/amazon_home_and_kitchen/SASRec/user_item_scores.csv.gz"
POPULARITY_SCORES_PATH = "deepseek_flash_output/amazon_home_and_kitchen/home_and_kitchen_k5_full_final_ranking.csv"
OUT_DIR = "deepseek_flash_output/amazon_home_and_kitchen"

PERSONALIZED_SCORE_COLS = ["sasrec_score"]
DEFAULT_BETAS = "0,0.05,0.1,0.3,0.5,0.7,0.9,1.0"
ID_COLS = ["user_id", "item_id", "original_item_id"]


def min_max_normalize_global(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Min-max normalize `score_col` across the ENTIRE dataset (not per user).
    If the whole column is constant (max == min), returns 0.5 for all rows
    (degenerate case, shouldn't happen in practice with real score spread).
    """
    col_min = df[score_col].min()
    col_max = df[score_col].max()
    denom = col_max - col_min
    if denom == 0:
        return pd.Series(0.5, index=df.index)
    return (df[score_col] - col_min) / denom


def load_personalized_scores(sasrec_path: str) -> pd.DataFrame:
    """Load the raw personalized model score export.
    """
    sasrec = pd.read_csv(sasrec_path, compression="gzip")
    print(f"[load] SASRec: {len(sasrec)} rows, {sasrec['user_id'].nunique()} users")
    return sasrec


def join_popularity(
    user_item_scores: pd.DataFrame,
    popularity_scores: pd.DataFrame,
    popularity_col: str,
    popularity_join_col: str = "original_item_id",
    drop_missing_popularity: bool = False,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Join key: popularity_scores' `item_id` corresponds to
    user_item_scores'  `popularity_join_col`.

    For Kitchen this is `original_item_id` (the catalog id happens to
    live there). For Baby/Douban (converted with convert_npz_to_long.py), `original_item_id` instead holds the internal
    MODEL item id (tie-break key only) -- the real catalog join key is
    `item_id` there. Pass --popularity_join_col item_id for those datasets."""
    before = len(user_item_scores)
    pop_renamed = popularity_scores[["item_id", popularity_col]].rename(
        columns={"item_id": popularity_join_col}
    )

    if drop_missing_popularity:
        # old behavior: drop items with no usable popularity score BEFORE
        # beta is ever applied -- shrinks the candidate pool for every beta,
        # including beta=1 where popularity shouldn't matter at all.
        pop_renamed = pop_renamed.dropna(subset=[popularity_col])
        merged = user_item_scores.merge(pop_renamed, on=popularity_join_col, how="inner")
        dropped = before - len(merged)
        if dropped > 0:
            print(f"[warn] dropped {dropped} / {before} rows with no usable popularity score "
                  f"for '{popularity_col}' (--drop_missing_popularity)")
    else:
        # default: keep every row; items with no popularity score get
        # y_hat_F = fill_value instead of being removed from the pool.
        merged = user_item_scores.merge(pop_renamed, on=popularity_join_col, how="left")
        n_missing = merged[popularity_col].isna().sum()
        if n_missing:
            print(f"[info] {n_missing} / {len(merged)} rows have no popularity score for "
                  f"'{popularity_col}' -- filled with {fill_value} (full candidate pool kept)")
        merged[popularity_col] = merged[popularity_col].fillna(fill_value)

    return merged


def iter_combined_scores(
    merged: pd.DataFrame,
    personalized_cols: list,
    popularity_col: str,
    betas: list,
):
    """Yields (model_name, beta, out_df) one (model, beta) combination at a
    time, instead of materializing every combination in memory at once --
    with many betas this cartesian product gets huge fast, easily OOMing
    even on a beefy box.
    """
    for model_col in personalized_cols:
        model_name = model_col.replace("_score", "")
        s_norm = min_max_normalize_global(merged, model_col)
        for beta in betas:
            s_new = beta * s_norm + (1 - beta) * merged[popularity_col]
            out = pd.DataFrame({
                "user_id": merged["user_id"],
                "item_id": merged["item_id"],
                "original_item_id": merged["original_item_id"],
                "model": model_name,
                "beta": beta,
                "s_personalized_norm": s_norm,
                "y_hat_F": merged[popularity_col],
                "s_new": s_new,
            })
            del s_new
            # tie policy matches the baseline: within a user, ties broken
            # by ascending original_item_id (not insertion order)
            out = out.sort_values(
                ["user_id", "s_new", "original_item_id"],
                ascending=[True, False, True],
                kind="mergesort",  # stable, so equal (user,s_new,orig_id) keep input order
            )
            out["rank_in_user"] = out.groupby("user_id").cumcount() + 1
            yield model_name, beta, out
            del out
            gc.collect()


def parse_args():
    p = argparse.ArgumentParser(description="Combine personalized + popularity scores (PARE-style).")
    p.add_argument("--sasrec_scores_path", default=SASREC_SCORES_PATH)
    p.add_argument("--popularity_scores_path", default=POPULARITY_SCORES_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument(
        "--popularity_col",
        required=True,
        choices=["norm_predict_popularity_score", "norm_rank10", "norm_rank20"],
        help="Which column in popularity_scores to use as y_hat_F",
    )
    p.add_argument(
        "--betas",
        default=DEFAULT_BETAS,
        help="Comma-separated beta values, e.g. '0,0.05,0.1,0.2,0.3,0.5,1.0'",
    )
    p.add_argument(
        "--drop_missing_popularity",
        action="store_true",
        help="Old behavior: inner-join + dropna on popularity_col instead of filling missing "
             "values with --fill_value. Shrinks the candidate pool for every beta.",
    )
    p.add_argument("--fill_value", type=float, default=0.0,
                    help="y_hat_F for items with no popularity score, when not dropping them.")
    p.add_argument(
        "--popularity_join_col", default="original_item_id",
        help="Column in the SASRec score file to join popularity_scores' item_id "
             "against. Kitchen: 'original_item_id' (default). Baby/Douban (converted via "
             "convert_npz_to_long.py): 'item_id' -- their 'original_item_id' is the internal "
             "model item id (tie-break only), not the catalog id.",
    )
    p.add_argument(
        "--dataset_name", default="kitchen",
        help="Used only to name output files (combined_<dataset_name>_pare_..._beta_X.csv.gz), "
             "so Baby/Douban runs don't get silently named 'kitchen'.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    betas = [float(b) for b in args.betas.split(",")]

    user_item_scores = load_personalized_scores(args.sasrec_scores_path)
    popularity_scores = pd.read_csv(args.popularity_scores_path)

    merged = join_popularity(
        user_item_scores,
        popularity_scores,
        popularity_col=args.popularity_col,
        popularity_join_col=args.popularity_join_col,
        drop_missing_popularity=args.drop_missing_popularity,
        fill_value=args.fill_value,
    )
    del user_item_scores, popularity_scores
    gc.collect()

    # one output file PER BETA -- not one giant file with every beta
    # stacked together. With many betas the single-file version balloons
    # into tens of GB on disk, and s24_evaluate_hr_ndcg_2.py loads a
    # combined file fully into memory (pd.read_csv), so a single
    # 500M-row file risks OOM even on a big node. One file per beta keeps
    # each read to a manageable size, and a failed/interrupted beta
    # doesn't take the rest down with it.
    def out_path_for(beta):
        return f"{args.out_dir}/combined_{args.dataset_name}_pare_{args.popularity_col}_beta_{beta}.csv.gz"

    header_written = {}
    rows_per_file = {}
    seen_users_per_file = {}
    total_rows = 0
    seen_models, seen_betas = set(), set()

    for model_name, beta, out in iter_combined_scores(
        merged, PERSONALIZED_SCORE_COLS, args.popularity_col, betas
    ):
        path = out_path_for(beta)
        print(f"[chunk] model={model_name} beta={beta}: {len(out)} rows, "
              f"{out['user_id'].nunique()} users -> {path}")
        out.to_csv(path, mode="a", header=not header_written.get(beta, False),
                   index=False, compression="gzip")
        header_written[beta] = True
        total_rows += len(out)
        rows_per_file[beta] = rows_per_file.get(beta, 0) + len(out)
        seen_users_per_file.setdefault(beta, set()).update(out["user_id"].unique())
        seen_models.add(model_name)
        seen_betas.add(beta)

    print(f"\ntotal rows: {total_rows}, models: {sorted(seen_models)}, betas: {sorted(seen_betas)}")
    print("saved files:")
    for beta in sorted(seen_betas):
        print(f"  beta={beta}: {rows_per_file[beta]} rows, "
              f"{len(seen_users_per_file[beta])} users -> {out_path_for(beta)}")
