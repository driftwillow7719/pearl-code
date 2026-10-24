"""
s13_merging_api_calls.py -- join feature_selection_mapping.csv (the
per-item query features + ground truth, produced when the JSONL batch was
prepared) with the LLM's batch results (selected_features, reasoning,
etc., produced by s12_batch_pipeline.py's `download` command), on
custom_id.

This gives you one row per item with everything: query features, ground
truth, category, embed_sim__* columns, AND the LLM's selected_features /
reasoning -- ready for the evaluation / 3x3 analysis scripts.

USAGE:
    python s13_merging_api_calls.py \
        --mapping feature_selection_mapping.csv \
        --results out/feature_selection.csv \
        --output feature_selection_merged.csv
"""
import argparse
import glob
import os

import pandas as pd


def resolve_result_paths(results_args):
    """results_args is a list (argparse nargs='+'). Each entry may itself
    be a glob pattern. Returns a flat, deduped, sorted list of concrete
    file paths."""
    paths = []
    for arg in results_args:
        if any(ch in arg for ch in "*?["):
            matched = sorted(glob.glob(arg))
            if not matched:
                raise FileNotFoundError(f"No files matched pattern: {arg}")
            paths.extend(matched)
        else:
            paths.append(arg)
    # dedupe while preserving order
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def dedupe_custom_id(df, label=""):
    dupes = df["custom_id"].duplicated().sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate custom_id(s) in {label} -- keeping first occurrence.")
        df = df.drop_duplicates(subset="custom_id", keep="first")
    return df


def load_results(results_args):
    """Loads and concatenates every results file resolved from
    results_args (list of paths and/or glob patterns) into one DataFrame."""
    paths = resolve_result_paths(results_args)
    print(f"Concatenating {len(paths)} results file(s):")
    for p in paths:
        print(f"  {p}")
    dfs = [pd.read_csv(p) for p in paths]
    results_df = pd.concat(dfs, ignore_index=True)
    return dedupe_custom_id(results_df, label="combined results")


def _check_no_overlap(mapping_df, results_df):
    # avoid clobbering identically-named columns from both sides (only
    # custom_id should overlap) -- surface a clear error rather than
    # silently suffixing if something unexpected collides
    shared_cols = (set(mapping_df.columns) & set(results_df.columns)) - {"custom_id"}
    if shared_cols:
        raise ValueError(
            f"Unexpected overlapping columns between mapping and results "
            f"(other than custom_id): {shared_cols} -- rename before merging "
            f"so nothing gets silently overwritten."
        )


def _do_merge(mapping_df, results_df, how, label=""):
    _check_no_overlap(mapping_df, results_df)
    merged = mapping_df.merge(results_df, on="custom_id", how=how, indicator=True)

    matched = (merged["_merge"] == "both").sum()
    mapping_only = (merged["_merge"] == "left_only").sum()
    results_only = (merged["_merge"] == "right_only").sum()
    print(f"{label}Merged: {matched} matched, {mapping_only} mapping-only (no result found), {results_only} results-only (no mapping row)")

    if how == "left" and mapping_only:
        print(f"{label}WARNING: {mapping_only} mapping row(s) had no matching result -- check for failed/missing batch requests.")

    return merged.drop(columns=["_merge"])


def merge_mapping_with_results(mapping_path, results_args, output_path, how="left"):
    """Combines every file in results_args into one DataFrame first, then
    merges once against the mapping CSV -- use this when the parts are
    non-overlapping shards of the same dataset (e.g. part1/part2 from a
    split full-dataset batch)."""
    mapping_df = pd.read_csv(mapping_path)
    results_df = load_results(results_args)

    print(f"mapping:  {len(mapping_df)} rows ({mapping_path})")
    print(f"results:  {len(results_df)} rows (combined from {len(results_args)} input(s))")

    merged = _do_merge(mapping_df, results_df, how)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged file ({len(merged)} rows) -> {output_path}")
    return merged


def merge_each_separately(mapping_path, results_args, output_dir, how="inner"):
    """Merges each results file in results_args individually against the
    full mapping CSV, writing one output file per input. Uses an INNER
    join by default, since each part only covers a subset of the full
    mapping -- a left join here would otherwise pad every output with a
    huge number of unmatched NaN rows for items that belong to the OTHER
    part(s)."""
    mapping_df = pd.read_csv(mapping_path)
    paths = resolve_result_paths(results_args)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    for p in paths:
        results_df = pd.read_csv(p)
        results_df = dedupe_custom_id(results_df, label=os.path.basename(p))

        base = os.path.basename(p)
        if base.endswith(".csv"):
            base = base[: -len(".csv")]
        out_path = os.path.join(output_dir, f"{base}_merged.csv")

        print(f"\n--- {p} ({len(results_df)} rows) ---")
        merged = _do_merge(mapping_df, results_df, how, label="  ")
        merged.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
        outputs.append(out_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Merge feature_selection_mapping.csv with batch results on custom_id.")
    parser.add_argument("--mapping", required=True, help="Path to feature_selection_mapping.csv")
    parser.add_argument("--results", required=True, nargs="+", help="One or more batch results CSV paths and/or glob patterns")
    parser.add_argument("--output", default=None, help="Path to write the combined merged CSV (default mode)")
    parser.add_argument("--output-dir", default=None, help="Directory to write one merged file per --results input (used with --separate)")
    parser.add_argument("--how", default=None, choices=["left", "inner", "outer"], help="Merge type (default: 'left' in combined mode, 'inner' in --separate mode)")
    parser.add_argument("--separate", action="store_true", help="Merge each --results file individually against the mapping, instead of combining them first")
    args = parser.parse_args()

    if args.separate:
        if not args.output_dir:
            parser.error("--separate requires --output-dir")
        merge_each_separately(args.mapping, args.results, args.output_dir, how=args.how or "inner")
    else:
        if not args.output:
            parser.error("--output is required unless --separate is used")
        merge_mapping_with_results(args.mapping, args.results, args.output, how=args.how or "left")


if __name__ == "__main__":
    main()