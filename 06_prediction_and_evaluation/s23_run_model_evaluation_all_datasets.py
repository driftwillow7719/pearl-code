"""
Unified evaluation script: Douban + Amazon (baby / home_and_kitchen).

Per-dataset behaviour
---------------------
douban
    * candidate pool filtered to imdb ids listed in --meta-csv (db_meta.csv)
    * Rate <= 0 remapped to 3
    * interactions split by time: everything before --douban-start is
      "history" (used only to remove already-seen movies from each user's
      candidate list), [--douban-start, --douban-end) is the ground truth
    * only users with interactions inside the window are evaluated
amazon (baby / home_and_kitchen)
    * exactly as before: the interaction file IS the test-month ground truth,
      no date filtering, no seen-item exclusion, no imdb filtering

NDCG
----
Uses TRAIL's formula (binary relevance Rate > 0, IDCG over ALL of the user's
relevant items in the window, result divided by the user's interaction count).
It is defined locally in this file so it does not depend on which version of
metric_measurement_order_utils.py is on disk. It must receive ONLY the single
user's interactions -- never the whole interaction table.

--rank-by predict        (default) rank items by predict_popularity_score
--rank-by ground_truth   ground-truth experiment: rank by ground_truth_popularity_score

Examples
--------
    # Douban, from mapping + prediction files
    python3 s23_run_model_evaluation_all_datasets.py --category douban \\
        --results-dir deepseek_flash_output/douban_movies/run_3 \\
        --variant-tag k10_full \\
        --meta-csv datasets/douban_test/db_meta.csv \\
        --interaction-csv datasets/douban_test/douban_movie_review.csv

    # Douban, from an already merged CSV, ground-truth experiment
    python3 s23_run_model_evaluation_all_datasets.py --category douban \\
        --merged-csv douban_baseline_1.csv --rank-by ground_truth \\
        --meta-csv db_meta.csv --interaction-csv douban_movie_review.csv

    # Amazon
    python3 s23_run_model_evaluation_all_datasets.py --category baby \\
        --results-dir llama_2_7b_output/amazon_baby/run_1_1 \\
        --variant-tag baby_k1_full --interaction-csv Baby_test.csv
"""
import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from metric_measurement_order_utils import compute_map_at_k, compute_mrr_at_k
from metric_measurement_pointwise_utils import compute_mae_error, compute_mse_metric, compute_r2_score
from metric_measurement_topk_utils import compute_f1_at_k, compute_hit_ratio_at_k, compute_precision_at_k, compute_recall_at_k

plt.rcParams['font.size'] = 12

SCORE_COLUMNS = {
    "predict": "predict_popularity_score",
    "ground_truth": "ground_truth_popularity_score",
}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── NDCG (TRAIL formula, one user at a time) ─────────────────────────────

def compute_ndcg_at_k(inferenced_df: pd.DataFrame,
                      user_interact_df: pd.DataFrame,
                      K: int,
                      score_col: str = "predict_popularity_score") -> float:
    """
    NDCG@K for ONE user, same formula as TRAIL:
      - binary relevance (Rate > 0)
      - IDCG over all of the user's relevant items in the window
      - final value divided by the number of the user's interaction rows

    inferenced_df    : the user's ranked top-k list (movieid + score columns)
    user_interact_df : ONLY this user's interactions in the evaluation window
    score_col        : column that defines the ranking
    """
    inf = inferenced_df.copy()
    usr = user_interact_df.copy()

    # unify column names (Douban: MovieID/Rate, Amazon: item/rating)
    if "movieid" not in usr.columns:
        for c in ("MovieID", "item", "item_id"):
            if c in usr.columns:
                usr = usr.rename(columns={c: "movieid"})
                break
    if "Rate" not in usr.columns and "rating" in usr.columns:
        usr = usr.rename(columns={"rating": "Rate"})
    if "movieid" not in inf.columns and "item_id" in inf.columns:
        inf = inf.rename(columns={"item_id": "movieid"})

    if len(usr) == 0:
        return 0.0

    # align id dtypes: int if possible (matches TRAIL), otherwise string
    try:
        inf["movieid"] = inf["movieid"].astype(int)
        usr["movieid"] = usr["movieid"].astype(int)
    except (ValueError, TypeError):
        inf["movieid"] = inf["movieid"].astype(str).str.strip()
        usr["movieid"] = usr["movieid"].astype(str).str.strip()

    df = pd.merge(inf[["movieid", score_col]], usr[["movieid", "Rate"]],
                  on="movieid", how="outer").reset_index(drop=True)
    df[score_col] = df[score_col].fillna(-np.inf)
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(0.0)
    rel = (df["Rate"] > 0).astype(float).to_numpy()

    discounts = np.log2(np.arange(2, K + 2))

    # DCG: rank by score (ties by movieid)
    order = df.sort_values(by=[score_col, "movieid"], ascending=[False, True]).index
    rel_dcg = rel[order][:K]
    dcg = float(np.sum(rel_dcg / discounts[:len(rel_dcg)]))

    # IDCG: all relevant items of this user first
    order_i = df.sort_values(by=["Rate", "movieid"], ascending=[False, True]).index
    rel_idcg = (df.loc[order_i, "Rate"] > 0).astype(float).to_numpy()[:K]
    idcg = float(np.sum(rel_idcg / discounts[:len(rel_idcg)]))

    if idcg == 0.0:
        return 0.0
    return (dcg / idcg) / len(usr)


# ── loading ──────────────────────────────────────────────────────────────

def load_interaction_df(interaction_path: str) -> pd.DataFrame:
    """Amazon: header'd CSV (user, item, rating, time), or the headerless
    space-delimited PARE _tra_test.txt (Home & Kitchen). Douban: plain CSV."""
    if str(interaction_path).endswith(".txt"):
        df = pd.read_csv(
            interaction_path, sep=' ', header=None,
            names=['user', 'item', 'rating', 'time'],
        )
        print(f"Loaded raw _tra_test.txt interaction file: {len(df)} rows, "
              f"{df['user'].nunique()} distinct users.")
    else:
        df = pd.read_csv(interaction_path)
    return df


def _parse_prediction_content(content):
    """Raw LLM JSON string -> (predict_popularity_score, reasoning);
    (nan, None) if it is not usable."""
    if not isinstance(content, str) or not content.strip():
        return np.nan, None
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return np.nan, None
    if not isinstance(obj, dict):
        return np.nan, None
    return obj.get("predict_popularity_score", np.nan), obj.get("reasoning")


def load_merged_inference_df(mapping_path: str, pred_results_path: str) -> pd.DataFrame:
    """Left-joins mapping_<tag>.csv to test_pred_results_<tag>.csv on custom_id.
    Handles both raw-JSON (`content`) and already-parsed result files."""
    mapping_df = pd.read_csv(mapping_path)
    pred_df = pd.read_csv(pred_results_path)

    # An already-parsed predict_popularity_score column (e.g. GPT-4o-mini output)
    # takes precedence; otherwise parse the raw `content` JSON (vLLM output).
    if "predict_popularity_score" not in pred_df.columns and "content" in pred_df.columns:
        pred_cols = [c for c in ("custom_id", "content", "was_repaired", "repair_temperature") if c in pred_df.columns]
        merged = mapping_df.merge(pred_df[pred_cols], on="custom_id", how="left")

        n_unmatched = merged["content"].isna().sum()
        if n_unmatched:
            print(f"WARNING: {n_unmatched}/{len(merged)} mapping rows had no matching "
                  f"test_pred_results row for their custom_id (interrupted/incomplete "
                  f"run?) -- these will be dropped as missing predict_popularity_score.")

        parsed = merged["content"].apply(_parse_prediction_content)
        merged["predict_popularity_score"] = [p[0] for p in parsed]
        merged["reasoning"] = [p[1] for p in parsed]
    else:
        if "predict_popularity_score" not in pred_df.columns:
            raise ValueError(
                f"{pred_results_path} has neither a 'predict_popularity_score' column "
                f"nor a 'content' column -- got columns {pred_df.columns.tolist()}."
            )
        pred_cols = [c for c in ("custom_id", "predict_popularity_score", "reasoning", "was_repaired", "repair_temperature") if c in pred_df.columns]
        merged = mapping_df.merge(pred_df[pred_cols], on="custom_id", how="left")

        n_unmatched = merged["predict_popularity_score"].isna().sum()
        if n_unmatched:
            print(f"WARNING: {n_unmatched}/{len(merged)} mapping rows had no matching "
                  f"prediction for their custom_id -- these will be dropped as missing "
                  f"predict_popularity_score.")

    return merged


def clean_inferenced_dataframe(inferenced_df: pd.DataFrame) -> pd.DataFrame:
    inferenced_df = inferenced_df.copy()
    inferenced_df["predict_popularity_score"] = pd.to_numeric(inferenced_df["predict_popularity_score"], errors='coerce')
    cleaned = inferenced_df.dropna(subset=["predict_popularity_score"]).copy()
    cleaned["predict_popularity_score"] = cleaned["predict_popularity_score"].clip(lower=0)
    return cleaned


def plot_metric_bar_chart(metrics_dict, metric_name, title=None, save_path=None, dpi=600):
    """Generic helper, unused by default."""
    df = pd.DataFrame(metrics_dict)

    plt.figure(figsize=(3.2, 3.8))
    colors = sns.color_palette("tab10", n_colors=len(df['k']))

    sns.barplot(x='k', y=metric_name, data=df, palette=colors, width=1, edgecolor="black")
    plt.xlabel('Top K', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    if not title:
        title = f"{metric_name} vs TopK"
    plt.title(title, fontsize=14)

    plt.gca().set_xlim(-0.5, len(df['k']) - 0.5)
    plt.gca().set_xticks(range(len(df['k'])))
    plt.gca().set_xticklabels(df['k'], fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


# ── per-user metric rows (shared by both datasets) ───────────────────────

def _metric_rows_for_user(user_id, candidate_df, user_df, top_k_list, score_col):
    """candidate_df: ranked candidate list for this user with a `Rate` column.
    user_df: THIS user's interactions in the evaluation window."""
    rows = []
    for k in top_k_list:
        eval_df = candidate_df.head(k)
        rows.append({
            "user_id": user_id,
            "k": k,
            "precision_at_k": compute_precision_at_k(eval_df, k),
            "recall_at_k": compute_recall_at_k(eval_df, k),
            "f1_at_k": compute_f1_at_k(eval_df, k),
            "hit_ratio_at_k": compute_hit_ratio_at_k(eval_df, k),
            "ndcg_at_k": compute_ndcg_at_k(eval_df, user_df, k, score_col=score_col),
            "map_at_k": compute_map_at_k(eval_df, k),
            "mrr_at_k": compute_mrr_at_k(eval_df, k),
        })
    return rows


# ── Douban: time split + seen-movie exclusion ────────────────────────────

def userwise_metric_calculation_douban(inferenced_df: pd.DataFrame,
                                       interaction_df: pd.DataFrame,
                                       top_k_list,
                                       score_col: str,
                                       window_start: str,
                                       window_end: str,
                                       tz_name: str = "Asia/Shanghai") -> pd.DataFrame:
    interaction_df = interaction_df.copy()
    inferenced_df = inferenced_df.copy()
    interaction_df["MovieID"] = interaction_df["MovieID"].astype(str).str.strip()
    inferenced_df["movieid"] = inferenced_df["movieid"].astype(str).str.strip()
    interaction_df["Time"] = interaction_df["Time"].astype(int)

    # The month boundaries are interpreted in an EXPLICIT timezone so results
    # do not depend on the timezone of the machine running the script.
    tz = ZoneInfo(tz_name)
    start_unix = datetime.strptime(window_start, TIME_FORMAT).replace(tzinfo=tz).timestamp()
    end_unix = datetime.strptime(window_end, TIME_FORMAT).replace(tzinfo=tz).timestamp()

    history_df = interaction_df[interaction_df["Time"] < start_unix]
    current_df = interaction_df[(interaction_df["Time"] >= start_unix) &
                                (interaction_df["Time"] < end_unix)]
    seen_by_user = history_df.groupby("UserID")["MovieID"].apply(set).to_dict()

    global_rank_df = inferenced_df.sort_values(by=score_col, ascending=False)
    max_k = max(top_k_list)

    user_metrics = []
    for user_id, user_df in tqdm(current_df.groupby("UserID"),
                                 total=current_df["UserID"].nunique(), desc="Userwise Eval (douban)"):
        seen_movie_ids = seen_by_user.get(user_id, set())
        user_rate_map = dict(zip(user_df["MovieID"], user_df["Rate"]))

        candidates = global_rank_df[~global_rank_df["movieid"].isin(seen_movie_ids)].head(max_k).copy()
        candidates["Rate"] = candidates["movieid"].map(user_rate_map).fillna(0)

        user_metrics.extend(_metric_rows_for_user(user_id, candidates, user_df, top_k_list, score_col))

    return pd.DataFrame(user_metrics)


# ── Amazon: whole interaction file is the ground truth ───────────────────

def userwise_metric_calculation_amazon(inferenced_df: pd.DataFrame,
                                       interaction_df: pd.DataFrame,
                                       top_k_list,
                                       score_col: str) -> pd.DataFrame:
    """No date filtering and no seen-item exclusion: every user is ranked
    against the same full candidate pool."""
    interaction_df = interaction_df.copy()
    inferenced_df = inferenced_df.copy()

    interaction_df["item"] = interaction_df["item"].astype(str).str.strip()
    inferenced_df["movieid"] = inferenced_df["item_id"].astype(str).str.strip()

    global_rank_df = inferenced_df.sort_values(by=score_col, ascending=False)
    max_k = max(top_k_list)

    user_metrics = []
    for user_id, user_df in tqdm(interaction_df.groupby("user"),
                                 total=interaction_df["user"].nunique(), desc="Userwise Eval (amazon)"):
        user_rate_map = dict(zip(user_df["item"], user_df["rating"]))

        candidates = global_rank_df.head(max_k).copy()
        candidates["Rate"] = candidates["movieid"].map(user_rate_map).fillna(0)

        user_metrics.extend(_metric_rows_for_user(user_id, candidates, user_df, top_k_list, score_col))

    pointwise_metrics = {
        "mae_error": compute_mae_error(inferenced_df),
        "mse_error": compute_mse_metric(inferenced_df)[0],
        "rmse_error": compute_mse_metric(inferenced_df)[1],
        "r2_score": compute_r2_score(inferenced_df),
    }
    print("Overall pointwise metrics:", pointwise_metrics)

    return pd.DataFrame(user_metrics)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one inference run (Douban or Amazon) against its interaction test set."
    )
    parser.add_argument('--category',
                        choices=['douban', 'baby', 'home_and_kitchen'],
                        required=True)
    parser.add_argument('--results-dir', default=None,
                        help="Folder with mapping_<variant_tag>.csv and test_pred_results_<variant_tag>.csv.")
    parser.add_argument('--variant-tag', default=None,
                        help="Must match the variant_tag the run was produced with "
                             "(also used to name the output files).")
    parser.add_argument('--mapping-path', default=None,
                        help="Explicit mapping CSV (overrides <results-dir>/mapping_<variant-tag>.csv). "
                             "For custom runs whose files do not follow the naming convention.")
    parser.add_argument('--pred-results-path', default=None,
                        help="Explicit prediction-results CSV (overrides "
                             "<results-dir>/test_pred_results_<variant-tag>.csv). Raw-JSON `content` "
                             "files and already-parsed files (GPT-4o-mini) are both accepted.")
    parser.add_argument('--interaction-csv', required=True,
                        help="Douban: douban_movie_review.csv. Amazon: header'd test CSV "
                             "or the raw PARE _tra_test.txt (Home & Kitchen).")
    parser.add_argument('--top-k-list', default="5,10,20")
    parser.add_argument('--output-dir', default=None,
                        help="Defaults to --results-dir (or the merged CSV's folder).")
    parser.add_argument('--rank-by', choices=['predict', 'ground_truth'], default='predict',
                        help="'ground_truth' for ground-truth experiments.")

    # Douban only
    parser.add_argument('--meta-csv', default=None,
                        help="Douban only: db_meta.csv (column `imdb`) -- candidate whitelist.")
    parser.add_argument('--merged-csv', default=None,
                        help="Any category: an already merged inference CSV (mapping columns + "
                             "predict_popularity_score, or a raw `content` column); skips the "
                             "mapping/pred merge.")
    parser.add_argument('--douban-start', default="2019-11-01 00:00:00",
                        help="Douban only: start of the evaluation month (validation: 2019-10-01 00:00:00).")
    parser.add_argument('--douban-end', default="2019-12-01 00:00:00",
                        help="Douban only: end (exclusive) of the evaluation month (validation: 2019-11-01 00:00:00).")
    parser.add_argument('--douban-tz', default="Asia/Shanghai",
                        help="Douban only: timezone used to interpret --douban-start/--douban-end "
                             "(default Asia/Shanghai, UTC+8, same as China time).")
    args = parser.parse_args()

    is_douban = args.category == "douban"
    use_merged_csv = args.merged_csv is not None

    if is_douban and not args.meta_csv:
        parser.error("--meta-csv is required for --category douban")
    explicit_paths = bool(args.mapping_path and args.pred_results_path)
    if (args.mapping_path or args.pred_results_path) and not explicit_paths:
        parser.error("--mapping-path and --pred-results-path must be given together")
    if not use_merged_csv and not explicit_paths and not (args.results_dir and args.variant_tag):
        parser.error("give --results-dir with --variant-tag, or --mapping-path with "
                     "--pred-results-path (or --merged-csv for an already merged file)")

    top_k_list = [int(k) for k in args.top_k_list.split(",")]
    score_col = SCORE_COLUMNS[args.rank_by]

    if use_merged_csv:
        variant_tag = args.variant_tag or os.path.splitext(os.path.basename(args.merged_csv))[0]
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.merged_csv))
    else:
        pred_file = args.pred_results_path or ""
        variant_tag = args.variant_tag or os.path.splitext(os.path.basename(pred_file))[0]
        output_dir = (args.output_dir or args.results_dir
                      or os.path.dirname(os.path.abspath(pred_file)))
    os.makedirs(output_dir, exist_ok=True)

    # ---- load predictions ----
    if use_merged_csv:
        print(f"Loading pre-merged file: {args.merged_csv}")
        merged_df = pd.read_csv(args.merged_csv)
        if "predict_popularity_score" not in merged_df.columns and "content" in merged_df.columns:
            parsed = merged_df["content"].apply(_parse_prediction_content)
            merged_df["predict_popularity_score"] = [p[0] for p in parsed]
            merged_df["reasoning"] = [p[1] for p in parsed]
    else:
        mapping_path = args.mapping_path or f"{args.results_dir}/mapping_{variant_tag}.csv"
        pred_results_path = args.pred_results_path or f"{args.results_dir}/test_pred_results_{variant_tag}.csv"
        print(f"Loading + merging: {mapping_path}  +  {pred_results_path}")
        merged_df = load_merged_inference_df(mapping_path, pred_results_path)
        merged_path = f"{output_dir}/{variant_tag}_merged.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged inference+prediction data -> {merged_path}")

    cleaned_df = clean_inferenced_dataframe(merged_df)
    n_dropped = len(merged_df) - len(cleaned_df)
    if n_dropped:
        print(f"Dropped {n_dropped}/{len(merged_df)} rows with unparseable/missing predict_popularity_score.")

    interaction_df = load_interaction_df(args.interaction_csv)

    # ---- dataset-specific preparation + evaluation ----
    print(f"Starting evaluation... (category={args.category}, rank-by={args.rank_by})")
    if is_douban:
        if "movieid" not in cleaned_df.columns and "movie_id" in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={"movie_id": "movieid"})

        valid_imdb = set(pd.read_csv(args.meta_csv)["imdb"].tolist())
        cleaned_df = cleaned_df[cleaned_df["imdb"].isin(valid_imdb)]
        print(f"Candidate pool after valid_imdb filter: {cleaned_df.shape}")

        interaction_df["Rate"] = interaction_df["Rate"].astype(float)
        interaction_df.loc[interaction_df["Rate"] <= 0, "Rate"] = 3

        user_metrics_df = userwise_metric_calculation_douban(
            cleaned_df, interaction_df, top_k_list, score_col,
            window_start=args.douban_start, window_end=args.douban_end,
            tz_name=args.douban_tz,
        )
        id_col = "movieid"
    else:
        if "item_id" not in cleaned_df.columns:
            for alt in ("movieid", "movie_id"):
                if alt in cleaned_df.columns:
                    cleaned_df = cleaned_df.rename(columns={alt: "item_id"})
                    break
        user_metrics_df = userwise_metric_calculation_amazon(
            cleaned_df, interaction_df, top_k_list, score_col,
        )
        id_col = "item_id"

    user_metrics_path = f"{output_dir}/{variant_tag}_userwise_metrics.csv"
    user_metrics_df.to_csv(user_metrics_path, index=False)
    print(f"Saved per-user metrics -> {user_metrics_path}")

    metric_columns = [
        'precision_at_k', 'recall_at_k', 'f1_at_k', 'hit_ratio_at_k',
        'ndcg_at_k', 'map_at_k', 'mrr_at_k'
    ]
    macro_avg = user_metrics_df.groupby('k')[metric_columns].mean().reset_index()
    macro_avg_path = f"{output_dir}/{variant_tag}_userwise_metrics_average.csv"
    macro_avg.to_csv(macro_avg_path, index=False)
    print(f"Saved macro-averaged metrics -> {macro_avg_path}")
    print(macro_avg)

    # Sanity check: top-20 by prediction vs by ground truth
    df1 = cleaned_df.sort_values(by='predict_popularity_score', ascending=False)
    df2 = cleaned_df.sort_values(by='ground_truth_popularity_score', ascending=False)
    print("Top-20 items by predicted popularity:")
    print(df1.head(20)[[id_col, 'predict_popularity_score']])
    print("Top-20 items by ground-truth popularity:")
    print(df2.head(20)[[id_col, 'ground_truth_popularity_score']])


if __name__ == "__main__":
    main()