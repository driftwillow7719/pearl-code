"""
Amazon (baby / kitchen) and Douban datasets.

All of the actual analysis logic (parsing "pop_cluster__direction__bucket"
center strings back into (pop_cluster, direction), building the 3x3
ground-truth x retrieved direction confusion matrix overall and per
ground_truth_pop_cluster, per-class precision/recall/F1 with macro
averages, and the full GT-group x retrieved-group breakdown) is UNCHANGED
-- domain-agnostic, and parse_center_name already handles the
"{pop_cluster}__{direction}__{bucket}" shape s14_retrieve_top1_centers.py's
top_1_center values have (e.g. "high_popular__increase__pooled"), same as
the movie reference.

Run per category, e.g.:
    python s15_3x3_analysis.py --dataset baby --mode active
"""
import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = ["increase", "decrease", "stable"]
POP_CLUSTERS = ["high_popular", "medium_popular", "zero_popular"]


def build_full_group_matrix(df):
    """Full GT-group x Retrieved-group crosstab (using the same group
    keys as group_profiles.json, e.g. 'high_popular__increase',
    'medium_popular__stable') -- unlike build_matrix's 3x3, which only
    tracks retrieved DIRECTION within a fixed GT pop_cluster and discards
    which retrieved pop_cluster tier a prediction landed in, this keeps
    the full retrieved_top_1_center string. Lets you see e.g. whether a
    high_popular__increase item predicted as "stable" landed in
    high_popular__stable specifically, or got pushed down a tier into
    medium_popular__stable or zero_popular__stable -- the two are very
    different failure modes (wrong direction only, vs. wrong direction
    AND wrong tier).
    """
    valid = df[df["ground_truth_direction"].notna() & df["ground_truth_pop_cluster"].notna()
               & df["top_1_center"].notna()].copy()
    valid["ground_truth_group"] = valid["ground_truth_pop_cluster"] + "__" + valid["ground_truth_direction"]
    matrix = pd.crosstab(
        valid["ground_truth_group"],
        valid["top_1_center"],
        rownames=["Ground Truth group ->"],
        colnames=["Retrieved group v"],
    )
    return matrix, len(valid)


def print_group_breakdown(gt_group, matrix, top_n=None):
    """Prints one row of build_full_group_matrix's output for a single GT
    group (e.g. 'high_popular__increase'), sorted descending by count with
    row percentages -- shows exactly which retrieved group(s) items with
    this true label end up in, flagging any '__stable' destination so you
    can see which specific pop_cluster's stable bucket absorbs them.
    """
    if gt_group not in matrix.index:
        print(f"\nnote: no ground-truth rows found for group '{gt_group}'.")
        return None

    row = matrix.loc[gt_group]
    row = row[row > 0].sort_values(ascending=False)
    if top_n:
        row = row.head(top_n)
    total = matrix.loc[gt_group].sum()

    print(f"\n{'-'*60}")
    print(f"  Retrieved-group breakdown for GT group = '{gt_group}'  (n={total})")
    print(f"{'-'*60}")
    for retrieved_group, count in row.items():
        pct = count / total * 100 if total else 0
        # retrieved_group is the raw top_1_center string, e.g.
        # "high_popular__stable__pooled" (see module docstring: s14 always
        # scores against "__pooled" centers) -- checking the parsed
        # direction rather than a raw endswith("__stable") is what
        # actually catches this, since every value here ends in
        # "__pooled", not the direction itself.
        _, direction = parse_center_name(retrieved_group)
        flag = "  <-- stable" if direction == "stable" else ""
        print(f"    {retrieved_group:<28} {count:>5}  ({pct:5.1f}%){flag}")
    return row


def compute_per_class_metrics(matrix):
    """Precision/recall/F1 per class, plus macro-averaged versions.
    Macro metrics give each direction (increase/decrease/stable) equal
    weight regardless of how many ground-truth items fall in it -- raw
    accuracy rewards whichever setup is better at the majority class;
    macro metrics don't let that dominate."""
    values = matrix.values.astype(float)
    labels = matrix.index.tolist()  # == CATEGORIES, matching row/col order

    row_sums = values.sum(axis=1)   # actual count per true class
    col_sums = values.sum(axis=0)   # predicted count per class
    diag = np.diag(values)

    recall = np.divide(diag, row_sums, out=np.full_like(diag, np.nan), where=row_sums > 0)
    precision = np.divide(diag, col_sums, out=np.full_like(diag, np.nan), where=col_sums > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.full_like(diag, np.nan), where=(precision + recall) > 0
    )

    per_class = pd.DataFrame({
        "class": labels, "n_true": row_sums.astype(int),
        "precision": precision, "recall": recall, "f1": f1,
    })

    # nanmean skips a class with zero ground-truth instances in this
    # subset (e.g. zero_popular's 'increase' row, which can't be scored)
    macro = {
        "macro_precision": np.nanmean(precision),
        "macro_recall": np.nanmean(recall),   # == balanced accuracy
        "macro_f1": np.nanmean(f1),
    }
    return per_class, macro


def parse_center_name(center_name):
    """'high_popular__decrease' -> ('high_popular', 'decrease').
    'medium_popular__decrease__pooled' -> ('medium_popular', 'decrease')
    -- any trailing '__{bucket}' segment is discarded, since this function
    only ever needs pop_cluster/direction. Returns (None, None) for
    anything unparseable (e.g. a scoring error row)."""
    if not isinstance(center_name, str) or "__" not in center_name:
        return None, None
    parts = center_name.split("__")
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def build_eval_df(top1_df, top1_csv):
    """top1_df already has ground_truth_direction / ground_truth_pop_cluster
    on every row (attached upstream by s14) -- this just parses
    retrieved_pop_cluster / retrieved_direction off top_1_center."""
    for col in ("ground_truth_direction", "ground_truth_pop_cluster"):
        if col not in top1_df.columns:
            raise ValueError(
                f"'{col}' not found in {top1_csv} -- make sure s14_retrieve_top1_centers.py "
                f"attached ground truth before saving (see its attach_ground_truth_direction)."
            )

    eval_df = top1_df.copy()
    parsed = eval_df["top_1_center"].apply(parse_center_name)
    eval_df["retrieved_pop_cluster"] = parsed.apply(lambda t: t[0])
    eval_df["retrieved_direction"] = parsed.apply(lambda t: t[1])
    return eval_df


def build_matrix(df):
    valid = df[df["retrieved_direction"].notna() & df["ground_truth_direction"].notna()]
    matrix = pd.crosstab(
        valid["ground_truth_direction"],
        valid["retrieved_direction"],
        rownames=["Ground Truth ->"],
        colnames=["Retrieved v"],
    ).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    return matrix, len(valid)


def print_matrix(title, matrix, n_valid, n_total):
    print(f"\n{'='*60}")
    print(f"  {title}  (n={n_total}, scored={n_valid})")
    print(f"{'='*60}")

    if n_valid == 0:
        print("  (no scoreable rows in this subset)")
        return None, None

    print("\n  3x3 Matrix (rows=Ground Truth, cols=Retrieved)")
    print(matrix.to_string())

    row_totals = matrix.sum(axis=1)
    row_pct = matrix.div(row_totals.replace(0, np.nan), axis=0).round(3) * 100
    print("\n  Row % per GT direction")
    print(row_pct.to_string())

    accuracy = np.trace(matrix.values) / matrix.values.sum() if matrix.values.sum() else 0.0
    print(f"\n  Overall accuracy (diagonal / total): {accuracy:.1%}")

    per_class, macro = compute_per_class_metrics(matrix)
    print("\n  Per-class precision / recall / F1")
    print(per_class.to_string(
        index=False,
        formatters={c: (lambda v: f"{v:.1%}" if pd.notna(v) else "n/a") for c in ("precision", "recall", "f1")},
    ))
    print(f"\n  Macro-averaged recall (balanced accuracy): {macro['macro_recall']:.1%}")
    print(f"  Macro-averaged precision:                  {macro['macro_precision']:.1%}")
    print(f"  Macro-averaged F1:                          {macro['macro_f1']:.1%}")

    return per_class, macro


def main():
    parser = argparse.ArgumentParser(
        description="3x3 ground-truth x retrieved-direction confusion matrix "
                     "and metrics for one Amazon category's s14 output."
    )
    parser.add_argument('--dataset', choices=['baby','kitchen','douban'], required=True)
    parser.add_argument('--mode', choices=['active', 'inactive'], default='active',
                         help="Matches the mode suffix s14_retrieve_top1_centers.py "
                              "appended to its output filename.")
    parser.add_argument('--top1-csv', default=None,
                         help="Defaults to processed_results/{dataset}/"
                              "{dataset}_top_1_centers_{mode}.csv")
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to processed_results/{dataset}/eval_results/{mode}")
    parser.add_argument('--feature-csv', default='datasets/fs_results/G1/fs_test_merged.csv',
                        help="Douban only: merged feature-selection results CSV.")
    parser.add_argument('--douban-meta-csv', required=False, default=None,
                        help="Douban only: metadata CSV whose 'imdb' column lists the valid test items.")
    args = parser.parse_args()

    dataset = args.dataset
    top1_csv = args.top1_csv or f"processed_results/{dataset}/{dataset}_top_1_centers_{args.mode}.csv"
    output_dir = args.output_dir or f"processed_results/{dataset}/eval_results/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset == 'douban':
        top1_df = pd.read_csv(top1_csv)
        feature_df = pd.read_csv(args.feature_csv)
        merged_csv = pd.merge(feature_df,top1_df[['doc_id','top_1_center']],on='doc_id',how='left')
        print(merged_csv.shape)
        filter_imdb_df = pd.read_csv(args.douban_meta_csv)
        valid_imdb = set(filter_imdb_df["imdb"])
        final_top1_df = merged_csv[merged_csv["imdb"].isin(valid_imdb)].reset_index(drop=True)
        eval_df = build_eval_df(final_top1_df, top1_csv)
    else:
        top1_df = pd.read_csv(top1_csv)
        eval_df = build_eval_df(top1_df, top1_csv)

    n_errors = eval_df["top_1_center"].isna().sum() if "top_1_center" in eval_df.columns else 0
    if n_errors:
        print(f"note: {n_errors} rows had no top_1_center (scoring errors) -- excluded from matrices")

    summary_rows = []

    matrix, n_valid = build_matrix(eval_df)
    _, macro = print_matrix("OVERALL 3x3 Direction Analysis", matrix, n_valid, len(eval_df))
    matrix.to_csv(f"{output_dir}/3x3_matrix_overall.csv")
    if macro:
        accuracy = np.trace(matrix.values) / matrix.values.sum()
        summary_rows.append({"subset": "overall", "n": n_valid, "accuracy": accuracy, **macro})

    for pop_cluster in POP_CLUSTERS:
        subset = eval_df[eval_df["ground_truth_pop_cluster"] == pop_cluster]
        sub_matrix, sub_n_valid = build_matrix(subset)
        _, sub_macro = print_matrix(f"{pop_cluster.upper()} 3x3 Direction Analysis", sub_matrix, sub_n_valid, len(subset))
        sub_matrix.to_csv(f"{output_dir}/3x3_matrix_{pop_cluster}.csv")
        if sub_macro:
            sub_accuracy = np.trace(sub_matrix.values) / sub_matrix.values.sum() if sub_matrix.values.sum() else 0.0
            summary_rows.append({"subset": pop_cluster, "n": sub_n_valid, "accuracy": sub_accuracy, **sub_macro})

    pd.DataFrame(summary_rows).to_csv(f"{output_dir}/metrics_summary.csv", index=False)

    # ---- per-row detail, for spot-checking ----
    detail_cols = [
        "custom_id", "doc_id", "item_id", "category",
        "ground_truth_direction", "ground_truth_pop_cluster",
        "retrieved_direction", "retrieved_pop_cluster", "top_1_center", "top_1_score",
    ]
    eval_df[[c for c in detail_cols if c in eval_df.columns]].to_csv(
        f"{output_dir}/direction_detail.csv", index=False
    )
    print(f"\nSaved matrices and detail CSV to {output_dir}/")

    # ---- full GT-group x retrieved-group breakdown (keeps retrieved
    # pop_cluster tier, unlike the direction-only 3x3 matrices above) ----
    full_matrix, full_n = build_full_group_matrix(eval_df)
    full_matrix.to_csv(f"{output_dir}/full_group_matrix.csv")
    print(f"\nSaved full GT-group x retrieved-group matrix ({full_matrix.shape[0]}x"
          f"{full_matrix.shape[1]}, n={full_n}) to {output_dir}/full_group_matrix.csv")

    print_group_breakdown("high_popular__increase", full_matrix)
    print_group_breakdown("medium_popular__increase", full_matrix)


if __name__ == "__main__":
    main()
