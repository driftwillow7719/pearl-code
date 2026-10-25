"""
s111_batch_splitter.py -- split an oversized batch JSONL file into parts that
each stay under OpenAI's 200 MB batch input file limit.

Each part gets its own suffix appended to a base tag (e.g. "part1",
"part2", ...) so you can submit/track/download each one separately with
your existing s12_batch_pipeline.py, using a distinct --tag per part:

    python s12_batch_pipeline.py submit --tag full_set_part1 \
        --jsonl out/feature_selection_batch_part1.jsonl
    python s12_batch_pipeline.py submit --tag full_set_part2 \
        --jsonl out/feature_selection_batch_part2.jsonl
    ... (repeat for however many parts were created)

    python s12_batch_pipeline.py status --tag full_set_part1
    python s12_batch_pipeline.py download --tag full_set_part1 --wait
    ... (repeat per part)

Then concatenate the downloaded per-part result CSVs (see
concat_part_results() below, or just pd.concat + to_csv by hand) into one
final results file once every part has completed.

USAGE (as a script):
    python s111_batch_splitter.py \
        --input out/feature_selection_batch.jsonl \
        --output-dir out \
        --max-mb 150

--max-mb defaults to 150 (not 200) to leave headroom -- OpenAI's limit is
a hard 200 MB, and staying a bit under it avoids any edge-case rejection
from JSON re-encoding/whitespace differences between your file and
whatever OpenAI measures on their end.
"""
import argparse
import glob
import json
import os

import pandas as pd


def split_jsonl(input_path, output_dir=None, max_mb=150, base_tag=None):
    """
    Splits input_path into whole-line chunks, each under max_mb megabytes.
    Returns the list of output file paths written, in order.
    """
    output_dir = output_dir or os.path.dirname(input_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if base_tag is None:
        base_tag = os.path.basename(input_path)
        if base_tag.endswith(".jsonl"):
            base_tag = base_tag[: -len(".jsonl")]

    max_bytes = int(max_mb * 1024 * 1024)

    total_lines = 0
    total_bytes = 0
    part_paths = []
    part_idx = 1
    cur_bytes = 0
    cur_lines = 0
    cur_file = None

    def open_new_part():
        nonlocal cur_file, cur_bytes, cur_lines, part_idx
        path = os.path.join(output_dir, f"{base_tag}_part{part_idx}.jsonl")
        cur_file = open(path, "w", encoding="utf-8")
        part_paths.append(path)
        cur_bytes = 0
        cur_lines = 0

    open_new_part()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # sanity check: make sure this line is valid JSON with a
            # custom_id before we commit it to a part file -- catches a
            # corrupt/truncated line early rather than silently shipping it
            try:
                obj = json.loads(line)
                if "custom_id" not in obj:
                    raise ValueError("missing custom_id")
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Invalid JSONL line encountered while splitting "
                    f"(line {total_lines + 1}): {e}"
                )

            line_bytes = len(line.encode("utf-8")) + 1  # +1 for the newline

            if cur_lines > 0 and cur_bytes + line_bytes > max_bytes:
                cur_file.close()
                part_idx += 1
                open_new_part()

            cur_file.write(line + "\n")
            cur_bytes += line_bytes
            cur_lines += 1
            total_lines += 1
            total_bytes += line_bytes

    if cur_file is not None:
        cur_file.close()

    print(f"Input: {input_path} ({total_bytes / (1024*1024):.1f} MB, {total_lines} requests)")
    print(f"Split into {len(part_paths)} part(s), each <= {max_mb} MB:")
    for p in part_paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        with open(p, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"  {p}  ({size_mb:.1f} MB, {n_lines} requests)")

    return part_paths


def concat_part_results(pattern, output_path):
    """
    Convenience helper for AFTER all parts have been downloaded: pass a
    glob pattern matching each part's downloaded results CSV (e.g.
    'out/full_set_part*_results.csv') and this
    concatenates them into one final CSV, sorted by custom_id.

    Adjust the glob pattern / column name to match whatever
    s12_batch_pipeline.py's `download` command actually names its output
    columns as (this assumes a 'custom_id' column exists, consistent with
    every other script in this pipeline).
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    print(f"Concatenating {len(paths)} part result file(s):")
    for p in paths:
        print(f"  {p}")

    dfs = [pd.read_csv(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)

    if "custom_id" in combined.columns:
        dupes = combined["custom_id"].duplicated().sum()
        if dupes:
            print(f"WARNING: {dupes} duplicate custom_id(s) across parts -- check for overlap.")
        combined = combined.sort_values("custom_id").reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"Saved combined results ({len(combined)} rows) -> {output_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Split an oversized batch JSONL into parts under OpenAI's 200 MB limit.")
    parser.add_argument("--input", required=True, help="Path to the oversized .jsonl file")
    parser.add_argument("--output-dir", default=None, help="Where to write part files (default: same dir as --input)")
    parser.add_argument("--max-mb", type=float, default=150, help="Max size per part, in MB (default: 150; hard API limit is 200)")
    parser.add_argument("--base-tag", default=None, help="Base name for part files (default: derived from --input filename)")
    args = parser.parse_args()

    split_jsonl(args.input, output_dir=args.output_dir, max_mb=args.max_mb, base_tag=args.base_tag)


if __name__ == "__main__":
    main()
