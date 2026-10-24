#!/usr/bin/env python3
"""
top_pop_recommendation.py
TopPop baseline: non-personalised, no LLM, no retrieval.

Produces a *merged CSV* that plugs straight into the unified evaluation script
(s23_run_model_evaluation_all_datasets.py, --merged-csv), so TopPop is evaluated with
exactly the same Douban protocol as the LLM runs: imdb whitelist (db_meta.csv), Rate<=0 -> 3,
history/ground-truth time split, seen-movie exclusion, same NDCG/MAP/MRR code.

Scoring: for each item, AGG (default max) of its popularity over the last W months of
trend_array. The last point of trend_array is the month BEFORE the target month, so there
is no leakage. Items with no history get score -1 (bottom), not 0.

Ties (very common: many items at 0/1) are broken deterministically, no randomness:
  1) score  2) latest value  3) sum over window  4) sum over the full history
  5) (only if still identical) original row order.
predict_popularity_score = the TopPop score itself. The output file is written in TopPop rank order.

Required id columns per category:
  douban                    : movie_id AND imdb  (+ --meta-csv db_meta.csv, filtered BEFORE ranking)
  baby / home_and_kitchen   : item_id            (no filtering, same as the Amazon evaluation)

Usage:
  python3 top_pop_recommendation.py --category douban --test datasets/261006_douban_full_test_set.csv \
      --windows 3 --agg max --meta-csv datasets/douban_test/db_meta.csv --out-prefix toppop_douban
  -> toppop_douban_W3_merged.csv      (use --windows all for the full history)
  python3 top_pop_recommendation.py --category baby --test <baby_test>.csv --windows 3 --out-prefix toppop_baby

Then evaluate (same as any other Douban run):
  python3 s23_run_model_evaluation_all_datasets.py --category douban \
      --merged-csv toppop_douban_W3_merged.csv --variant-tag toppop_W3 \
      --meta-csv datasets/douban_test/db_meta.csv \
      --interaction-csv datasets/douban_test/douban_movie_review.csv
"""
import argparse, re
import numpy as np
import pandas as pd

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_trend(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    if not isinstance(x, str):
        return np.array([], dtype=float)
    return np.array([float(t) for t in NUM.findall(x)], dtype=float)


def score_item(tr, w, agg):
    if len(tr) == 0 or (len(tr) == 1 and tr[0] == -1):
        return -1.0, -1.0, -1.0, -1.0     # no history
    win = tr if w is None else tr[-w:]      # w=None -> 'all' = full history
    s = {"max": win.max(), "mean": win.mean(), "sum": win.sum()}[agg]
    return float(s), float(tr[-1]), float(win.sum()), float(tr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["douban", "baby", "home_and_kitchen"], required=True)
    ap.add_argument("--test", required=True, help="full test set CSV (needs trend_array and "
                                                   "ground_truth_popularity_score, plus the id columns of the category)")
    ap.add_argument("--windows", default="3", help="comma list, e.g. 1,3,6,all ('all' = whole history; one merged CSV per window)")
    ap.add_argument("--agg", choices=["max", "mean", "sum"], default="max")
    ap.add_argument("--trend-col", default="trend_array")
    ap.add_argument("--meta-csv", default=None,
                    help="Douban only: db_meta.csv (column `imdb`); keep ONLY these items BEFORE ranking")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.test).reset_index(drop=True)
    is_douban = a.category == "douban"
    need = ["ground_truth_popularity_score", a.trend_col]
    need += ["movie_id", "imdb"] if is_douban else ["item_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] --category {a.category}: test file is missing column(s) {missing}")

    if is_douban:
        if not a.meta_csv:
            raise SystemExit("[error] --category douban needs --meta-csv (db_meta.csv) so ranking is done "
                             "only among the whitelisted items")
        valid_imdb = set(pd.read_csv(a.meta_csv)["imdb"].astype(str))
        n0 = len(df)
        df = df[df["imdb"].astype(str).isin(valid_imdb)].reset_index(drop=True)
        print(f"[filter] imdb in {a.meta_csv}: {n0} -> {len(df)} items (ranking is done among these only)")
    elif a.meta_csv:
        print("[note] --meta-csv is Douban-only and is ignored for Amazon categories.")
    trends = [parse_trend(x) for x in df[a.trend_col]]
    n = len(df)

    for tok in [x.strip() for x in a.windows.split(",")]:
        w = None if tok.lower() == "all" else int(tok)
        sc = np.array([score_item(t, w, a.agg) for t in trends])
        order = np.lexsort((-sc[:, 3], -sc[:, 2], -sc[:, 1], -sc[:, 0]))   # last key = primary; lexsort is stable
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(1, n + 1)

        out = df.copy()
        out["predict_popularity_score"] = sc[:, 0]        # the TopPop score itself
        out["toppop_rank"] = rank
        out = out.sort_values("toppop_rank").reset_index(drop=True)   # file is in TopPop rank order
        path = f"{a.out_prefix}_W{tok}_merged.csv"
        out.to_csv(path, index=False)
        print(f"[W={tok}, agg={a.agg}] {n} items | no-history: {int((sc[:, 0] < 0).sum())} -> {path}")


if __name__ == "__main__":
    main()