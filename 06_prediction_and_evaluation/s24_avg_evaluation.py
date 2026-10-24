"""
Average hit_ratio_at_5/10 and ndcg_at_5/10 across the 3 seeded replicate
runs (run_1/run_2/run_3), per model x dataset x retrieval variant.

Reads each run's per-variant evaluation output, named:
    {dataset_name}_{exp_set}_userwise_metrics_average.csv
e.g. baby_k5_full_userwise_metrics_average.csv, douban_no_retrieval_userwise_metrics_average.csv

Each such file has one row per top-k cutoff (k=5,10,20) with columns
precision_at_k, recall_at_k, f1_at_k, hit_ratio_at_k, ndcg_at_k, map_at_k,
mrr_at_k -- this script only pulls hit_ratio_at_k/ndcg_at_k for k=5 and
k=10, then averages each across however many of the 3 runs are found.

Folder layout expected (adjust MODEL_OUTPUT_DIRS / DATASETS / RUNS below if
yours differs):
    <model_output_dir>/<dataset_subdir>/run_1/{dataset_name}_{exp_set}_userwise_metrics_average.csv
    <model_output_dir>/<dataset_subdir>/run_2/...
    <model_output_dir>/<dataset_subdir>/run_3/...

A missing run/file is skipped with a WARNING (not an error) and averaged
over however many runs ARE present -- useful while some jobs are still
in progress -- and the output records n_runs so you can see at a glance
which rows are based on fewer than 3 runs.

Run:
    python3 s24_avg_evaluation.py
Output:
    averaged_ranking_metrics_3runs.csv (one row per model x dataset x exp_set)
"""
from pathlib import Path

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
# key -> folder name actually used on disk for that model's output
MODEL_OUTPUT_DIRS = {
    # "llama2_7b": "llama_2_7b_output",
    # "qwen2_7b": "qwen2_7b_output",
    "qwen2_7b_ins": "qwen2_7b_ins_output",
    "llama3_8b": "llama_3_8b_output",
    # "gpt4o_mini": "gpt4omini_output",
    # "deepseek_flash": "deepseek_flash_output"
}

# key -> (dataset subfolder under <model_output_dir>/, dataset_name used in the CSV filename)
DATASETS = {
    "baby": ("amazon_baby", "baby"),
    "douban": ("douban_movies", "douban"),
     "home_and_kitchen": ("amazon_home_and_kitchen", "home_and_kitchen"),

}

EXP_SETS = ["no_retrieval", "k1_full", "k5_full", "k10_full"]
RUNS = ["run_1", "run_2", "run_3"]
K_VALUES_WANTED = [5, 10]
METRIC_COLS = ["hit_ratio_at_k", "ndcg_at_k"]

# One output CSV per model x dataset (not one combined file). Written here
# by default (current directory) -- change to e.g. "averaged_metrics" to
# collect them in their own subfolder instead.
OUTPUT_DIR = "."
# ── END CONFIG ────────────────────────────────────────────────────────────


def load_run_metrics(csv_path: Path) -> dict:
    """Read one run's *_userwise_metrics_average.csv and return
    {(k, metric_col): value} for k in K_VALUES_WANTED."""
    df = pd.read_csv(csv_path)
    out = {}
    for k in K_VALUES_WANTED:
        row = df[df["k"] == k]
        if row.empty:
            print(f"WARNING: {csv_path} has no row for k={k} -- skipping that k for this run")
            continue
        for col in METRIC_COLS:
            out[(k, col)] = row.iloc[0][col]
    return out


def average_one_variant(model_dir: str, dataset_subdir: str, dataset_name: str, exp_set: str) -> dict | None:
    per_run_values = []
    n_found = 0
    for run in RUNS:
        csv_path = Path(model_dir) / dataset_subdir / run / f"{dataset_name}_{exp_set}_userwise_metrics_average.csv"
        if not csv_path.is_file():
            print(f"WARNING: missing {csv_path} -- skipping this run for this variant")
            continue
        per_run_values.append(load_run_metrics(csv_path))
        n_found += 1

    if not per_run_values:
        print(f"WARNING: no runs at all found for {model_dir}/{dataset_subdir}/{exp_set} -- skipping this row entirely")
        return None

    row = {"n_runs": n_found}
    for k in K_VALUES_WANTED:
        for col in METRIC_COLS:
            vals = [rv[(k, col)] for rv in per_run_values if (k, col) in rv]
            out_col = f"{col.replace('_at_k', '')}_at_{k}"  # e.g. hit_ratio_at_5
            avg_val = sum(vals) / len(vals)
            row[out_col] = round(avg_val,6) if vals else None
    return row


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    written = []

    for model_key, model_dir in MODEL_OUTPUT_DIRS.items():
        for dataset_key, (dataset_subdir, dataset_name) in DATASETS.items():
            rows = []
            for exp_set in EXP_SETS:
                row = average_one_variant(model_dir, dataset_subdir, dataset_name, exp_set)
                if row is None:
                    continue
                rows.append({"exp_set": exp_set, **row})

            if not rows:
                print(f"WARNING: no data at all for {model_key}/{dataset_key} -- no file written")
                continue

            df = pd.DataFrame(rows)
            out_path = Path(OUTPUT_DIR) / f"averaged_ranking_metrics_{model_key}_{dataset_key}.csv"
            df.to_csv(out_path, index=False)
            written.append(out_path)
            print(f"\n=== {model_key} / {dataset_key} -> {out_path} ===")
            print(df.to_string(index=False))

    print(f"\nWrote {len(written)} file(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()