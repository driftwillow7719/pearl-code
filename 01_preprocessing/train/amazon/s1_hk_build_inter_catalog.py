"""
Create home_and_kitchen_inter_train.csv-style item catalog (item_id, inter_month_count,
begin_month, max_month) from PARE's own pop_train.csv / pop_valid.csv / pop_test.csv
output.

Usage
-----
    python s1_hk_build_inter_catalog.py \\
        --train      reviews_Home_and_Kitchen_5_pop_train.csv \\
        --valid      reviews_Home_and_Kitchen_5_pop_valid.csv \\
        --test       reviews_Home_and_Kitchen_5_pop_test.csv \\
        --info-file  reviews_Home_and_Kitchen_5_infos_pop.txt \\
        --out        Home_and_Kitchen_inter_train.csv
"""
import argparse
import ast
import os

import pandas as pd


_EMPTY_COLS = ["item", "time_release", "time", "pop_history", "pop_gt", "valid_pop_len"]


def load_idx_to_asin_map(info_path):
    """Read PARE's own {dataset}_infos_pop.txt and return {integer_id: asin}.

    Line index 2 (0-indexed, i.e. the 3rd line) of that file is
    `str(dict_item_idx)`, a Python dict literal `{asin: integer_id}` --
    see loaded_json() in data_set.py: `config.dict_item_idx =
    eval(lines[2].strip())`. We read it the same way (eval, since it's a
    plain dict/int/str literal written by str() on a dict -- not attacker
    input, and ast.literal_eval handles this fine too) and invert it,
    since we need integer_id -> asin, not asin -> integer_id.
    """
    with open(info_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError(
            f"{info_path} has only {len(lines)} line(s) -- expected at least 3 "
            f"(num_item, max_time, dict_item_idx on line index 2). Is this really "
            f"a PARE *_infos_pop.txt file?"
        )
    dict_item_idx = ast.literal_eval(lines[2].strip())
    if not isinstance(dict_item_idx, dict):
        raise ValueError(
            f"Line index 2 of {info_path} did not parse to a dict (got "
            f"{type(dict_item_idx)}) -- expected {{asin: integer_id}}."
        )
    idx_to_asin = {v: k for k, v in dict_item_idx.items()}
    if len(idx_to_asin) != len(dict_item_idx):
        raise ValueError(
            f"{info_path}'s dict_item_idx has {len(dict_item_idx)} entries but "
            f"only {len(idx_to_asin)} unique integer IDs -- can't invert it "
            f"cleanly (this shouldn't happen for a real PARE output file)."
        )
    return idx_to_asin


def _load_split(path, split_name):
    if path is None:
        return pd.DataFrame(columns=_EMPTY_COLS + ["_split"])
    try:
        if os.path.getsize(path) == 0:
            raise pd.errors.EmptyDataError
        # dtype={"item": str} is critical -- some asins are all-digit
        # (e.g. "0076144011"), and without this pandas silently infers that
        # column as int64, which also STRIPS LEADING ZEROS. That corrupts
        # the item id invisibly (no error), and later breaks joins against
        # any string-keyed source (like the metadata gzip file).
        df = pd.read_csv(path, dtype={"item": str})
    except pd.errors.EmptyDataError:
        # A genuinely empty split (0 items had any activity in that window)
        # is a real, valid outcome -- not a bug -- so this is a no-op, not
        # an error.
        print(f"  note: {path} has no rows (empty {split_name} split)")
        return pd.DataFrame(columns=_EMPTY_COLS + ["_split"])
    if df.empty:
        df["_split"] = split_name
        return df
    df["pop_history"] = df["pop_history"].apply(ast.literal_eval)
    df["_split"] = split_name
    return df


def build_inter_catalog(train_path, valid_path=None, test_path=None, info_path=None):
    train = _load_split(train_path, "train")
    valid = _load_split(valid_path, "valid")
    test = _load_split(test_path, "test")

    if train.empty:
        raise ValueError("train_path produced zero rows -- inter_month_count can't be "
                          "reconstructed without it (it defines the item universe).")

    # Translate PARE's internal integer item IDs back into real asins BEFORE
    # any grouping/dedup logic runs, so everything downstream (item_universe,
    # the final 'item' output column) is keyed by the real asin. Without
    # this, the 'item' column is meaningless integers that will 0%-match
    # any asin-keyed metadata file.
    if info_path is not None:
        idx_to_asin = load_idx_to_asin_map(info_path)
        for df in (train, valid, test):
            if df.empty:
                continue
            # item column was read as str (dtype={"item": str}) to dodge the
            # leading-zero int64 coercion bug, but it holds an *integer* ID
            # here, so cast back to int to look it up in idx_to_asin.
            translated = df["item"].astype(int).map(idx_to_asin)
            n_unmapped = translated.isna().sum()
            if n_unmapped:
                raise ValueError(
                    f"{n_unmapped} row(s) have an integer item ID with no entry in "
                    f"{info_path}'s dict_item_idx -- info file doesn't match these "
                    f"split CSVs (wrong dataset/run?)."
                )
            df["item"] = translated
    else:
        print("  WARNING: no --info-file given -- the 'item' column will be left as "
              "PARE's raw internal integer IDs, NOT real asins. This will NOT match "
              "against an asin-keyed metadata file. Pass --info-file to fix this.")

    all_rows = pd.concat([train, valid, test], ignore_index=True)

    # Global timeline length: the largest `time` seen anywhere across all three
    # splits. This is PARE's own max_time (the test row's `time`, when a test
    # split is present and non-empty).
    max_time = int(all_rows["time"].max())

    # Item universe = everything in train (== PARE's dict_item_idx pool).
    item_universe = train["item"].unique()

    rows = []
    for item in item_universe:
        item_rows = all_rows[all_rows["item"] == item]
        # Pick the row with the largest `time` for this item -- test beats
        # valid beats train at the same `time` (shouldn't tie in practice
        # since the three splits cover disjoint month ranges).
        split_priority = {"test": 2, "valid": 1, "train": 0}
        item_rows = item_rows.assign(_prio=item_rows["_split"].map(split_priority))
        latest = item_rows.sort_values(["time", "_prio"], ascending=[False, False]).iloc[0]

        time_release = int(latest["time_release"])
        valid_pop_len = int(latest["valid_pop_len"])
        pop_gt = int(latest["pop_gt"])
        latest_time = int(latest["time"])

        # True history length should always be (latest_time - time_release).
        # PARE's own data_set.py has a special case: when the target month IS
        # the release month, the real pop_history slice is empty, but PARE
        # substitutes a fake placeholder [0] and reports valid_pop_len=1
        # instead of the true 0 (`if len(pop_history) == 0: pop_history = [0]`).
        # Trust the *arithmetic* (latest_time - time_release), not the stored
        # valid_pop_len, whenever they disagree -- that placeholder is not
        # real historical data and must not be inserted into the sequence.
        true_hist_len = latest_time - time_release
        if true_hist_len <= 0:
            pop_history = []
        else:
            pop_history = latest["pop_history"][:valid_pop_len]
            assert len(pop_history) == true_hist_len, (
                f"item {item}: unexpected valid_pop_len mismatch "
                f"({valid_pop_len} vs expected {true_hist_len}) outside the known "
                f"release-month placeholder case -- worth investigating directly."
            )

        # [0]*time_release  ->  months before release
        # pop_history + [pop_gt]  ->  release month through latest_time (real data)
        # zero-fill  ->  latest_time+1 .. max_time (assumption, see docstring)
        real_part = pop_history + [pop_gt]
        n_assumed_zero_tail = max_time - latest_time  # 0 if this item's latest row IS the global max_time
        inter_month_count = (
            [0] * time_release
            + real_part
            + [0] * n_assumed_zero_tail
        )

        assert len(inter_month_count) == max_time + 1, (
            f"item {item}: length mismatch ({len(inter_month_count)} vs {max_time + 1}); "
            f"time_release={time_release}, valid_pop_len={valid_pop_len}, "
            f"latest_time={latest_time}, max_time={max_time}"
        )

        begin_month = time_release
        max_month = int(pd.Series(inter_month_count).idxmax())

        rows.append({
            "item": item,
            "inter_month_count": inter_month_count,
            "begin_month": begin_month,
            "max_month": max_month,
            "latest_row_split": latest["_split"],
            "latest_row_time": latest_time,
            "assumed_zero_tail_months": n_assumed_zero_tail,
        })

    out = pd.DataFrame(rows)
    out.insert(0, "item_id", range(len(out)))

    n_with_assumption = (out["assumed_zero_tail_months"] > 0).sum()
    print(f"Built catalog for {len(out)} items. Global timeline length: {max_time + 1} months.")
    print(f"Items whose latest row is a TRAIN row not reaching global max_time "
          f"(zero-tail assumption applied): {n_with_assumption} / {len(out)} "
          f"({n_with_assumption / len(out):.1%})")
    if n_with_assumption:
        print("  -> these items' final months are assumed 0, not measured. See "
              "'assumed_zero_tail_months' column to inspect/exclude them if needed.")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="...pop_train.csv")
    ap.add_argument("--valid", default=None, help="...pop_valid.csv (optional)")
    ap.add_argument("--test", default=None, help="...pop_test.csv (optional)")
    ap.add_argument("--info-file", default=None,
                     help="PARE's {dataset}_infos_pop.txt -- REQUIRED to get real asins in "
                          "the 'item' column instead of PARE's internal integer IDs. "
                          "Without this, the output 'item' column will NOT match against "
                          "any asin-keyed metadata file.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = build_inter_catalog(a.train, a.valid, a.test, a.info_file)
    # Match Baby_inter_train.csv's stored-as-string-list convention for the list column.
    out_to_save = out.copy()
    out_to_save["inter_month_count"] = out_to_save["inter_month_count"].apply(str)
    out_to_save.to_csv(a.out, index=False)
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()