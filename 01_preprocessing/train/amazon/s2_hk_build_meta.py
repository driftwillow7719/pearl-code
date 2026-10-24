"""
Convert meta_<Category>.json.gz (Amazon's per-category metadata file, the
same format PARE's data_set.py reads for side_info) into a Baby_meta.csv-style CSV: one row per item, with the exact fields
amazon_inter_synthetic_data_generation.py's format_item_description() /
build_descriptions_from_meta() expect: item_id, title, description, brand,
salesRank, categories, price (+ the raw asin, kept as 'item').

Items are NOT dropped for missing title/description/categories. This matches
Baby's original behavior: every item in the inter catalog gets a row
here, with any missing field left blank (NaN) and rendered as "" downstream
by format_item_description(). A per-field missingness report is still
printed for visibility, but it is diagnostic only -- it does not filter rows.

Usage
-----
    python s2_hk_build_meta.py \\
        --meta-gz meta_Home_and_Kitchen.json.gz \\
        --inter-catalog Home_and_Kitchen_inter_train.csv \\
        --out Home_and_Kitchen_meta.csv
"""
import argparse
import ast
import gzip

import pandas as pd

KEEP_FIELDS = ["asin", "title", "description", "brand", "salesRank", "categories", "price"]
REPORT_FIELDS = ["title", "description", "categories"]  # diagnostic-only, not filtered on


def load_raw_meta(meta_gz_path):
    """One row per line in the gzip file, keeping only the fields the
    downstream description-formatting code actually uses."""
    rows = []
    with gzip.open(meta_gz_path, "rb") as g:
        for line in g:
            d = ast.literal_eval(line.decode("utf-8") if isinstance(line, bytes) else line)
            rows.append({k: d.get(k) for k in KEEP_FIELDS})
    df = pd.DataFrame(rows)
    df = df.rename(columns={"asin": "item"})
    return df


def _is_blank_string_field(val):
    """NaN, or empty/whitespace-only after stripping -> True (missing)."""
    if pd.isna(val):
        return True
    return str(val).strip() == ""


def _categories_is_present(categories_raw):
    """categories comes through as a Python list literal already (e.g.
    "[['Home & Kitchen', 'Kitchen & Dining']]"), NOT a string needing
    ast.literal_eval a second time here -- it was already evaluated once by
    load_raw_meta's ast.literal_eval on the whole line. So this just checks
    for NaN / an empty list, matching format_item_description's own
    parse_category() semantics (empty list == missing)."""
    if categories_raw is None or (isinstance(categories_raw, float) and pd.isna(categories_raw)):
        return False
    try:
        return len(categories_raw) > 0
    except TypeError:
        return False


def meta_row_report_flags(row):
    """Per-field present/missing flags for title/description/categories --
    used only for the diagnostic print in build_meta_csv(), never to drop rows."""
    return {
        "title": not _is_blank_string_field(row.get("title")),
        "description": not _is_blank_string_field(row.get("description")),
        "categories": _categories_is_present(row.get("categories")),
    }


def build_meta_csv(meta_gz_path, inter_catalog_path):
    raw_meta = load_raw_meta(meta_gz_path)
    n_dup_asins = raw_meta['item'].duplicated().sum()
    print(f"Duplicate asins in raw metadata file: {n_dup_asins}")
    # dtype={"item": str} for the same reason as s1_hk_build_inter_catalog.py:
    # all-digit asins get silently misread as int64 (dropping leading zeros)
    # otherwise, which then fails to match raw_meta's string-typed asin below.
    catalog = pd.read_csv(inter_catalog_path, usecols=["item_id", "item"], dtype={"item": str})

    before = len(catalog)
    merged = catalog.merge(raw_meta, on="item", how="left")
    n_no_match = merged[KEEP_FIELDS[1:]].isna().all(axis=1).sum()  # no metadata at all matched

    print(f"Catalog items: {before}")
    print(f"Items with a metadata match: {before - n_no_match} ({(before - n_no_match) / before:.1%})")
    if n_no_match:
        print(f"Items with NO metadata match at all: {n_no_match} "
              f"(likely items with an asin present in reviews but absent from the "
              f"category's metadata dump -- these get an empty description "
              f"downstream, same as everything else missing a field)")

    # --- diagnostic-only report (NOT a filter): per-field missingness on title/description/categories ---
    flags = merged.apply(meta_row_report_flags, axis=1, result_type="expand")
    for field in REPORT_FIELDS:
        n_missing = int((~flags[field]).sum())
        print(f"  {field} missing: {n_missing} / {before} ({n_missing / before:.1%})")
    n_all_three_missing_at_least_one = int((~flags.all(axis=1)).sum())
    print(f"  missing >=1 of title/description/categories: "
          f"{n_all_three_missing_at_least_one} / {before} "
          f"({n_all_three_missing_at_least_one / before:.1%}) -- NOT dropped, kept with blank fields")

    cols = ["item", "categories", "description", "title", "price", "brand", "salesRank", "item_id"]
    return merged[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-gz", required=True)
    ap.add_argument("--inter-catalog", required=True,
                     help="Output of s1_hk_build_inter_catalog.py -- supplies the item_id mapping.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    

    out = build_meta_csv(a.meta_gz, a.inter_catalog)
    out.to_csv(a.out, index=False)
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()