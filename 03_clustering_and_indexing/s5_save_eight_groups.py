"""
Amazon (baby / home_and_kitchen) and Douban: save eight groups.

Everything about HOW the 8 (pop_cluster, pop_direction) groups are built,
stratified by history-depth bucket, fallback-pooled for small cells, and
narrated style is domain-agnostic (it only ever talks about "popularity value", "item age",
"trend").

1. popularity_threshold's CUTOFFS depending on dataset selection.
2. The embedding SOURCE COLUMN is 'item_description' (Amazon product
   description, built earlier in this pipeline from Baby_meta.csv /
   Home_and_Kitchen_meta.csv while Douban is from db_meta.csv).
"""
import argparse
import json
import math
import re
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cosine
from dataset_config import DATASETS

BUCKET_ORDER = ('debut', 'recent_single_point', 'recent_multi_point', 'maturing')

# MATURING_TREND_POINT_THRESHOLD set at item_age = 3
MATURING_TREND_POINT_THRESHOLD = 3

# To mark small groups
DEFAULT_MIN_CELL_N = 15

_POP_HIGH = None   # set once in __main__ from the chosen dataset

# ======== POP CLUSTER THRESHOLD ==========
def popularity_threshold(pop_value):
    """Zero / medium / high. 'high' starts at DATASETS[...]['high']."""
    if pop_value == 0:
        return "zero_popular"
    elif pop_value > 0 and pop_value < _POP_HIGH:        
        return "medium_popular"
    else:
        return "high_popular"
    
# ======== RETRIEVED DIRECTION ==========
def get_retrieved_direction(item_age, popularity_value, ground_truth_score) -> str:
    if int(item_age) == -1:
        if ground_truth_score > 0:
            return 'increase'
        else:
            return 'stable'
    else:
        if ground_truth_score > popularity_value:
            return 'increase'
        elif ground_truth_score < popularity_value:
            return 'decrease'
        else:
            return 'stable'

# ======== AGE BUCKET ==========
def get_history_depth_bucket(item_age):
    """Returns the HISTORY DEPTH BUCKET (debut / recent_single_point /
    recent_multi_point / maturing) from item_age. Unchanged from the movie
    reference -- item_age's convention (-1 = debut / 0 = single historical
    point / N = N months of real history) is identical across this whole
    Amazon pipeline (amazon_inter_synthetic_data_generation.py,
    s5_preprocess_datasets_full.py), so no adaptation is needed here."""
    if item_age == -1:
        return "debut"
    elif item_age == 0:
        return 'recent_single_point'
    elif item_age <= MATURING_TREND_POINT_THRESHOLD:
        return 'recent_multi_point'
    else:
        return 'maturing'

# ======== LATEST MOMENTUM AND OTHER TREND RELATED STATS ==========
def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        # an item with < 2 real trend points has no trajectory to compute a
        # rate-of-change from -- "unavailable" (not "unknown") because this
        # isn't a data gap, it's a structural non-existence: a single point
        # can never have a momentum, by definition
        return "unavailable"

    arr = np.asarray(trend_array, dtype=int)
    w = min(window, len(arr) - 1)
    # average log1p difference over the trailing window, not just the last step
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w + 1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0


def trend_summary_stats(trend_array):
    arr = np.asarray(trend_array, dtype=int) if isinstance(trend_array, (list, np.ndarray)) else None
    # sentinel (no real history) or unusable input -- both an EMPTY array
    # (this pipeline's convention for debut, see
    # amazon_inter_synthetic_data_generation.py / s5_preprocess_datasets_full.py)
    # and a single [-1] sentinel (the movie reference's convention) are
    # treated identically here, so this function works unchanged regardless
    # of which "no history" encoding upstream code used.
    if arr is None or len(arr) == 0 or (len(arr) == 1 and arr[0] == -1):
        return {
            'trend_min': np.nan,
            'trend_max': np.nan,
            'trend_volatility': np.nan,
            'trend_overall_change': np.nan,
            'time_since_peak': np.nan,
        }

    trend_min = float(arr.min())
    trend_max = float(arr.max())

    # volatility and overall_change are only meaningful with >=2 points --
    # NaN (unmeasurable), not 0 (measured-and-flat), for a single real point
    trend_volatility = float(arr.std()) if len(arr) > 1 else np.nan
    trend_overall_change = float(arr[-1] - arr[0]) if len(arr) > 1 else np.nan

    # time_since_peak is genuinely well-defined even with one point --
    # that point trivially is both the peak and the most recent value
    peak_idx = int(arr.argmax())
    time_since_peak = (len(arr) - 1) - peak_idx  # 0 = peak is the most recent point

    return {
        'trend_min': trend_min,
        'trend_max': trend_max,
        'trend_volatility': trend_volatility,
        'trend_overall_change': trend_overall_change,
        'time_since_peak': time_since_peak,
    }

def _is_single_real_trend_point(trend_array):
    """
    True only if trend_array holds exactly one GENUINE historical
    observation -- not the -1 "no history at all" sentinel (movie
    convention) or an empty array (this pipeline's convention), and not
    missing/empty.
    """
    if not isinstance(trend_array, (list, np.ndarray)):
        return False
    arr = np.asarray(trend_array)
    if len(arr) != 1:
        return False
    return not (arr[0] == -1)

# ---------------------------------------------------------------------------
# Amazon-specific: parsing this pipeline's `historical_trend` string column
# into the plain numeric trend_array these functions expect. Not present in
# the movie reference (which got trend_array from a different upstream
# column) -- added here since it's needed before build_group_characteristics
# can run on Amazon training data at all.
# ---------------------------------------------------------------------------

_TREND_PAIR_RE = re.compile(r'\d{4}-\d{2}:(-?\d+(?:\.\d+)?)')

def parse_historical_trend_to_array(trend_str):
    """'2014-01:1, 2014-02:1, 2014-03:2' -> np.array([1., 1., 2.])
    'No historical trend' (or NaN/empty) -> np.array([])  (empty -- this
    pipeline's debut/no-history convention, see trend_summary_stats above
    for why an empty array is handled identically to the movie reference's
    [-1] sentinel)."""
    if pd.isna(trend_str):
        return np.array([])
    s = str(trend_str).strip()
    if s == '' or s == 'No historical trend':
        return np.array([])
    values = [float(v) for v in _TREND_PAIR_RE.findall(s)]
    return np.array(values, dtype=int)

# ---------------------------------------------------------------------------
# per-bucket-type stats blocks -- deliberately different field sets, see
# module docstring's "WHAT EACH BUCKET CARRIES" in the original v5 reference
# ---------------------------------------------------------------------------

def _single_point_stats_block(cell):
    """recent_single_point (or its fallback source): latest_pop_value +
    gt_pop_change only. No trend_min/max/volatility/overall_change/
    time_since_peak/latest_momentum -- all undefined or trivial for a
    single real trend point."""
    return {
        'n_items': len(cell),
        'latest_pop_value_min': cell['latest_pop_value'].min(),
        'latest_pop_value_max': cell['latest_pop_value'].max(),
        'latest_pop_value_25th': cell['latest_pop_value'].quantile(0.25),
        'latest_pop_value_median': cell['latest_pop_value'].median(),
        'latest_pop_value_75th': cell['latest_pop_value'].quantile(0.75),
        'gt_pop_change_min': cell['gt_pop_change'].min(),
        'gt_pop_change_25th': cell['gt_pop_change'].quantile(0.25),
        'gt_pop_change_median': cell['gt_pop_change'].median(),
        'gt_pop_change_75th': cell['gt_pop_change'].quantile(0.75),
        'gt_pop_change_max': cell['gt_pop_change'].max(),
    }

def _full_stats_block(cell, momentum_col='latest_momentum'):
    n_items = len(cell)

    momentum_numeric = pd.to_numeric(cell[momentum_col], errors='coerce')
    n_momentum_unavailable = int(momentum_numeric.isna().sum())
    n_single_trend_point = int(cell['trend_array'].apply(_is_single_real_trend_point).sum())

    multi_point_mask = ~cell['trend_array'].apply(_is_single_real_trend_point)
    peak_source = cell[multi_point_mask]
    n_multi_point_trend_items = int(multi_point_mask.sum())

    return {
        'n_items': n_items,
        'n_single_trend_point': n_single_trend_point,
        'n_multi_point_trend_items': n_multi_point_trend_items,
        'latest_pop_value_min': cell['latest_pop_value'].min(),
        'latest_pop_value_max': cell['latest_pop_value'].max(),
        'latest_pop_value_25th': cell['latest_pop_value'].quantile(0.25),
        'latest_pop_value_median': cell['latest_pop_value'].median(),
        'latest_pop_value_75th': cell['latest_pop_value'].quantile(0.75),
        'item_age_min': cell['item_age'].min(),
        'item_age_max': cell['item_age'].max(),
        'item_age_25th': cell['item_age'].quantile(0.25),
        'item_age_median': cell['item_age'].median(),
        'item_age_75th': cell['item_age'].quantile(0.75),
        'trend_min_min': peak_source['trend_min'].min(),
        'trend_min_25th': peak_source['trend_min'].quantile(0.25),
        'trend_min_median': peak_source['trend_min'].median(),
        'trend_min_75th': peak_source['trend_min'].quantile(0.75),
        'trend_min_max': peak_source['trend_min'].max(),
        'trend_max_min': peak_source['trend_max'].min(),
        'trend_max_25th': peak_source['trend_max'].quantile(0.25),
        'trend_max_median': peak_source['trend_max'].median(),
        'trend_max_75th': peak_source['trend_max'].quantile(0.75),
        'trend_max_max': peak_source['trend_max'].max(),
        'trend_volatility_min': cell['trend_volatility'].min(),
        'trend_volatility_25th': cell['trend_volatility'].quantile(0.25),
        'trend_volatility_median': cell['trend_volatility'].median(),
        'trend_volatility_75th': cell['trend_volatility'].quantile(0.75),
        'trend_volatility_max': cell['trend_volatility'].max(),
        'trend_overall_change_median': cell['trend_overall_change'].median(),
        'time_since_peak_median_multi_point': peak_source['time_since_peak'].median(),
        'time_since_peak_25th_multi_point': peak_source['time_since_peak'].quantile(0.25),
        'time_since_peak_75th_multi_point': peak_source['time_since_peak'].quantile(0.75),
        'latest_mom_min': momentum_numeric.min(),
        'latest_mom_25th': momentum_numeric.quantile(0.25),
        'latest_mom_median': momentum_numeric.median(),
        'latest_mom_75th': momentum_numeric.quantile(0.75),
        'latest_mom_max': momentum_numeric.max(),
        'latest_mom_unavailable_count': n_momentum_unavailable,
        'gt_pop_change_min': cell['gt_pop_change'].min(),
        'gt_pop_change_25th': cell['gt_pop_change'].quantile(0.25),
        'gt_pop_change_median': cell['gt_pop_change'].median(),
        'gt_pop_change_75th': cell['gt_pop_change'].quantile(0.75),
        'gt_pop_change_max': cell['gt_pop_change'].max(),
    }


# ---------------------------------------------------------------------------
# main builder (unchanged from v5)
# ---------------------------------------------------------------------------

def build_group_characteristics(dataset, embed, doc_id_col='doc_id', trend_col='trend_array',
                                 min_cell_n=DEFAULT_MIN_CELL_N):
    """
    dataset must already carry 'pop_cluster', 'pop_direction', and
    'history_depth_bucket' columns (see this module's __main__ for how to
    build them).

    Returns (results, non_debut_with_trends).

    results: dict keyed by (pop_cluster, pop_direction) ->
        {
          'buckets': {
              'debut':               {n_items, mean_embed, doc_ids}                      (only present if this group actually has >0 zero-real-trend items)
              'recent_single_point': {..latest_pop_value/gt_pop_change.., mean_embed, doc_ids, is_fallback, n_cell_items}
              'recent_multi_point':  {..full feature set.., mean_embed, doc_ids, is_fallback, n_cell_items}
              'maturing':            {..full feature set.., mean_embed, doc_ids, is_fallback, n_cell_items}
              # a bucket key is simply absent if the group has zero items in it
          },
          'pooled': {..full feature set.., over ALL non-debut buckets combined}  -- fallback
              source only, NOT narrated as its own section (see module docstring)
        }

    non_debut_with_trends: non-debut rows with trend stats + gt_pop_change
    columns added -- for reuse by correlation-study scripts etc.
    """
    non_debut = dataset[dataset['history_depth_bucket'] != 'debut'].copy()

    trend_stats_df = non_debut[trend_col].apply(trend_summary_stats).apply(pd.Series)
    non_debut_with_trends = pd.concat(
        [non_debut.reset_index(drop=True), trend_stats_df.reset_index(drop=True)], axis=1
    )
    non_debut_with_trends['gt_pop_change'] = (
        non_debut_with_trends['popularity_value'] - non_debut_with_trends['latest_pop_value']
    )
    non_debut_with_trends['n_trend_points'] = non_debut_with_trends[trend_col].apply(
        lambda arr: len(arr) if isinstance(arr, (list, np.ndarray)) else 0
    )
    non_debut_with_trends['is_single_real_trend_point'] = non_debut_with_trends[trend_col].apply(_is_single_real_trend_point)
    debut_df = dataset[dataset['history_depth_bucket'] == 'debut'].copy()

    all_groups = (
        set(non_debut_with_trends.groupby(['pop_cluster', 'pop_direction']).groups.keys())
        | set(debut_df.groupby(['pop_cluster', 'pop_direction']).groups.keys())
    )

    results = {}

    for (pop, direction) in sorted(all_groups):
        group_non_debut = non_debut_with_trends[
            (non_debut_with_trends['pop_cluster'] == pop) & (non_debut_with_trends['pop_direction'] == direction)
        ]
        group_debut = debut_df[(debut_df['pop_cluster'] == pop) & (debut_df['pop_direction'] == direction)]

        # pooled: fallback source ONLY (never narrated on its own) -- all
        # non-debut buckets combined for this group, using the multi-point
        # split since it mixes single- and multi-point items
        if len(group_non_debut) > 0:
            pooled_stats = _full_stats_block(group_non_debut)
            pooled_mean_embed = np.mean(embed[group_non_debut[doc_id_col].values], axis=0)
            pooled_stats['mean_embed'] = pooled_mean_embed
            pooled_stats['doc_ids'] = group_non_debut[doc_id_col].values
        else:
            pooled_stats, pooled_mean_embed = None, None

        buckets = {}

        # ---- debut: decided empirically (does this group have any items
        # with zero real trend points?), embed-only, no fallback
        # substitution ----
        if len(group_debut) > 0:
            debut_doc_indices = group_debut[doc_id_col].values
            buckets['debut'] = {
                'n_items': len(group_debut),
                'n_cell_items': len(group_debut),
                'is_fallback': False,
                'mean_embed': np.mean(embed[debut_doc_indices], axis=0),
                'doc_ids': debut_doc_indices,
            }

        # ---- recent_single_point / recent_multi_point / maturing ----
        for bucket in ('recent_single_point', 'recent_multi_point', 'maturing'):
            cell = group_non_debut[group_non_debut['history_depth_bucket'] == bucket]
            n_cell = len(cell)
            if n_cell == 0:
                continue

            is_fallback = n_cell < min_cell_n

            if is_fallback and pooled_stats is not None:
                if bucket == 'recent_single_point':
                    cell_stats = _single_point_stats_block(group_non_debut)
                else:
                    cell_stats = dict(pooled_stats)
                cell_mean_embed = pooled_mean_embed
                cell_doc_indices = group_non_debut[doc_id_col].values
            else:
                if bucket == 'recent_single_point':
                    cell_stats = _single_point_stats_block(cell)
                else:
                    cell_stats = _full_stats_block(cell)
                cell_mean_embed = np.mean(embed[cell[doc_id_col].values], axis=0)
                cell_doc_indices = cell[doc_id_col].values

            cell_stats['n_cell_items'] = n_cell
            cell_stats['is_fallback'] = is_fallback
            cell_stats['mean_embed'] = cell_mean_embed
            cell_stats['doc_ids'] = cell_doc_indices
            buckets[bucket] = cell_stats

        results[(pop, direction)] = {'buckets': buckets, 'pooled': pooled_stats}

    return results, non_debut_with_trends


def _narrate_pooled_group(stats, small_n_threshold=30):
    trend_min_phrase = (
        f"{_fmt(stats['trend_min_median'])} (middle 50%: {_fmt(stats['trend_min_25th'])} to "
        f"{_fmt(stats['trend_min_75th'])}; full range up to {_fmt(stats['trend_min_max'])})"
    )
    trend_max_phrase = (
        f"{_fmt(stats['trend_max_median'])} (middle 50%: {_fmt(stats['trend_max_25th'])} to "
        f"{_fmt(stats['trend_max_75th'])}; full range up to {_fmt(stats['trend_max_max'])})"
    )
    trend_volatility_phrase = (
        f"around {_fmt(stats['trend_volatility_median'])} (middle 50%: "
        f"{_fmt(stats['trend_volatility_25th'])} to {_fmt(stats['trend_volatility_75th'])}; "
        f"full range up to {_fmt(stats['trend_volatility_max'])})"
    )
    momentum_phrase = (
        f"typically {_fmt(stats['latest_mom_median'])} (middle 50%: "
        f"{_fmt(stats['latest_mom_25th'])} to {_fmt(stats['latest_mom_75th'])}; "
        f"full range: {_fmt(stats['latest_mom_min'])} to {_fmt(stats['latest_mom_max'])})"
    )

    return (
        f"Latest popularity value typically {_fmt(stats['latest_pop_value_median'])} "
        f"(middle 50%: {_fmt(stats['latest_pop_value_25th'])} to {_fmt(stats['latest_pop_value_75th'])}; "
        f"full range: {_fmt(stats['latest_pop_value_min'])} to {_fmt(stats['latest_pop_value_max'])}). "
        f"Item age typically {_fmt(stats['item_age_median'])} months (middle 50%: "
        f"{_fmt(stats['item_age_25th'])} to {_fmt(stats['item_age_75th'])}). "
        f"Popularity magnitude across the trend: typically as low as "
        f"{trend_min_phrase} and as high as {trend_max_phrase}. "
        f"Volatility {trend_volatility_phrase}, net change of "
        f"{_fmt(stats['trend_overall_change_median'])} from start to end of the trend. "
        f"{_peak_phrase(stats)} "
        f"Momentum {momentum_phrase}"
        f"{_momentum_unavailable_phrase(stats)}. "
        f"{_gt_pop_change_phrase(stats, small_n_threshold=small_n_threshold, noun='group')}"
    )


def report_bucket_cell_sizes(results):
    """(group, bucket) -> n table -- run this BEFORE trusting min_cell_n's
    default, especially for known-small groups like high_popular__stable."""
    rows = []
    for (pop, direction), group_result in results.items():
        for bucket in BUCKET_ORDER:
            cell = group_result['buckets'].get(bucket)
            if cell is None:
                rows.append({'group': f"{pop}__{direction}", 'bucket': bucket,
                              'n_cell_items': 0, 'is_fallback': 'n/a (no items)'})
            else:
                rows.append({'group': f"{pop}__{direction}", 'bucket': bucket,
                              'n_cell_items': cell['n_cell_items'], 'is_fallback': cell['is_fallback']})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def embed_scoring(embed, center_vector):
    """Cosine similarity between item embedding and a centroid, scaled to [0, 1]."""
    distance = cosine(embed, center_vector)
    similarity = 1 - distance
    normalized_sim = (1 + similarity) / 2
    return normalized_sim


def _scalar(value):
    """
    Coerces a group-characteristic value to a plain python float, safely.
    """
    if isinstance(value, pd.Series):
        if value.nunique() > 1:
            print(f"WARNING: expected a scalar but got a Series with "
                  f"DIFFERING values {value.tolist()} -- duplicate-column "
                  f"bug is likely corrupting this number, not just cosmetic. "
                  f"Fix the upstream duplicate columns before trusting this.")
        return float(value.iloc[0])
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def _group_key_to_str(key):
    """(pop_cluster, pop_direction) tuple -> safe string key for JSON/npz."""
    return f"{key[0]}__{key[1]}"


def _fmt(value, sig=3):
    """
    Formats a number to `sig` significant figures WITHOUT scientific
    notation. Returns "n/a" for NaN rather than crashing -- defensive, in
    case a caller ever passes a stats dict through the wrong narration path.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(sig - 1 - magnitude, 0)
    s = f"{value:.{decimals}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def _signed_fmt(value, sig=3):
    """Same as _fmt but always prefixes a sign (+/-)."""
    s = _fmt(value, sig)
    if s == "n/a":
        return s
    if value > 0 and not s.startswith('+'):
        return f"+{s}"
    return s


def _momentum_unavailable_phrase(stats, min_fraction_to_report=0.05):
    unavailable = stats.get('latest_mom_unavailable_count', 0)
    n_items = stats.get('n_items', 0)
    if n_items == 0 or unavailable / n_items < min_fraction_to_report:
        return ""
    available = n_items - unavailable
    return (f" (based on {available} of {n_items} items with available momentum -- "
            f"{unavailable} had momentum unavailable due to a limited trend pattern)")


def _gt_pop_change_phrase(stats, small_n_threshold=30, noun="bucket"):
    median = stats['gt_pop_change_median']
    p25 = stats['gt_pop_change_25th']
    p75 = stats['gt_pop_change_75th']
    vmin = stats['gt_pop_change_min']
    vmax = stats['gt_pop_change_max']

    small_n_note = ""
    if stats.get('n_items', 0) < small_n_threshold:
        small_n_note = " (small group -- treat this range as a rough indication only)"

    return (
        f"Historically, items in this {noun} changed by {_signed_fmt(median)} in "
        f"popularity value, from their pre-outcome (latest) value to the final observed "
        f"outcome (middle 50%: {_signed_fmt(p25)} to {_signed_fmt(p75)}; full range: "
        f"{_signed_fmt(vmin)} to {_signed_fmt(vmax)}){small_n_note}. This describes what "
        f"happened to labeled training items already known to belong to this {noun} -- it "
        f"is not computable for a new, unlabeled query item and must not be treated as one "
        f"of that item's own features."
    )


def _peak_phrase(stats):
    if stats.get('n_multi_point_trend_items', 0) > 0:
        return (
            f"Peak typically occurred {_fmt(stats['time_since_peak_median_multi_point'])} "
            f"months before the most recent observation (middle 50%: "
            f"{_fmt(stats['time_since_peak_25th_multi_point'])} to "
            f"{_fmt(stats['time_since_peak_75th_multi_point'])})."
        )
    return "No items in this bucket had more than one real historical trend point."


def narrate_group(key, group_result):
    """
    Narration for one (pop_cluster, direction) group, pooled across all
    non-debut history-depth buckets.
    """
    pop, direction = key
    buckets = group_result['buckets']
    pooled = group_result.get('pooled')

    debut_n = buckets['debut']['n_items'] if 'debut' in buckets else 0
    total_n = debut_n + (pooled['n_items'] if pooled else 0)

    if not pooled:
        return (
            f"Group: {pop} / {direction} (n=0 non-debut items). "
            f"{debut_n} of {total_n} items that historically reached this outcome were "
            f"debut items; no pooled trend/feature stats are available for this group."
        )

    body = _narrate_pooled_group(pooled)
    single_point_n = pooled.get('latest_mom_unavailable_count', 0)

    tail = (
        f"{debut_n} of {total_n} items that historically reached this outcome were debut "
        f"items (no popularity history at all at the time observed); {single_point_n} of "
        f"{pooled['n_items']} non-debut items in this group had only a single real "
        f"historical trend point -- useful context if the query item itself has little or "
        f"no trend history."
    )

    return f"Group: {pop} / {direction} (n={pooled['n_items']}). {body} {tail}"

def save_group_narrations_for_prompt(profiles, out_dir='.'):
    """Flat {group: narration} -- 8 entries, each already containing its
    bucket sub-sections."""
    narrations = {name: block['narration'] for name, block in profiles.items()}

    with open(f"{out_dir}/group_narrations_for_prompt.json", "w") as f:
        json.dump(narrations, f, indent=2)

    lines = []
    for name, narration in narrations.items():
        lines.append(f"### {name}")
        lines.append(narration)
        lines.append("")
    with open(f"{out_dir}/group_narrations_for_prompt.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {len(narrations)} group narrations (each with bucket sub-sections) to "
          f"{out_dir}/group_narrations_for_prompt.json and .txt (prompt-ready).")


def _clean_debut_stats(stats):
    return {
        'n_items': int(stats['n_items']),
        'n_cell_items': int(stats['n_cell_items']),
        'is_fallback': bool(stats['is_fallback']),
    }


def _clean_single_point_stats(stats):
    return {
        'n_items': int(stats['n_items']),
        'n_cell_items': int(stats['n_cell_items']),
        'is_fallback': bool(stats['is_fallback']),
        'latest_pop_value_min': _scalar(stats['latest_pop_value_min']),
        'latest_pop_value_max': _scalar(stats['latest_pop_value_max']),
        'latest_pop_value_25th': _scalar(stats['latest_pop_value_25th']),
        'latest_pop_value_median': _scalar(stats['latest_pop_value_median']),
        'latest_pop_value_75th': _scalar(stats['latest_pop_value_75th']),
        'gt_pop_change_min': _scalar(stats['gt_pop_change_min']),
        'gt_pop_change_25th': _scalar(stats['gt_pop_change_25th']),
        'gt_pop_change_median': _scalar(stats['gt_pop_change_median']),
        'gt_pop_change_75th': _scalar(stats['gt_pop_change_75th']),
        'gt_pop_change_max': _scalar(stats['gt_pop_change_max']),
    }

def _clean_full_stats(stats):
    cleaned = {
        'n_items': int(stats['n_items']),
        'n_single_trend_point': int(stats.get('n_single_trend_point', 0)),
        'n_multi_point_trend_items': int(stats.get('n_multi_point_trend_items', 0)),
        'latest_pop_value_min': _scalar(stats['latest_pop_value_min']),
        'latest_pop_value_max': _scalar(stats['latest_pop_value_max']),
        'latest_pop_value_25th': _scalar(stats['latest_pop_value_25th']),
        'latest_pop_value_median': _scalar(stats['latest_pop_value_median']),
        'latest_pop_value_75th': _scalar(stats['latest_pop_value_75th']),
        'item_age_min': _scalar(stats['item_age_min']),
        'item_age_max': _scalar(stats['item_age_max']),
        'item_age_25th': _scalar(stats['item_age_25th']),
        'item_age_75th': _scalar(stats['item_age_75th']),
        'item_age_median': _scalar(stats['item_age_median']),
        'trend_min_min': _scalar(stats['trend_min_min']),
        'trend_min_25th': _scalar(stats['trend_min_25th']),
        'trend_min_median': _scalar(stats['trend_min_median']),
        'trend_min_75th': _scalar(stats['trend_min_75th']),
        'trend_min_max': _scalar(stats['trend_min_max']),
        'trend_max_min': _scalar(stats['trend_max_min']),
        'trend_max_25th': _scalar(stats['trend_max_25th']),
        'trend_max_median': _scalar(stats['trend_max_median']),
        'trend_max_75th': _scalar(stats['trend_max_75th']),
        'trend_max_max': _scalar(stats['trend_max_max']),
        'trend_volatility_min': _scalar(stats['trend_volatility_min']),
        'trend_volatility_25th': _scalar(stats['trend_volatility_25th']),
        'trend_volatility_median': _scalar(stats['trend_volatility_median']),
        'trend_volatility_75th': _scalar(stats['trend_volatility_75th']),
        'trend_volatility_max': _scalar(stats['trend_volatility_max']),
        'trend_overall_change_median': _scalar(stats['trend_overall_change_median']),
        'time_since_peak_median_multi_point': _scalar(stats['time_since_peak_median_multi_point']),
        'time_since_peak_25th_multi_point': _scalar(stats['time_since_peak_25th_multi_point']),
        'time_since_peak_75th_multi_point': _scalar(stats['time_since_peak_75th_multi_point']),
        'latest_mom_min': _scalar(stats['latest_mom_min']),
        'latest_mom_25th': _scalar(stats['latest_mom_25th']),
        'latest_mom_median': _scalar(stats['latest_mom_median']),
        'latest_mom_75th': _scalar(stats['latest_mom_75th']),
        'latest_mom_max': _scalar(stats['latest_mom_max']),
        'latest_mom_unavailable_count': int(stats.get('latest_mom_unavailable_count', 0)),
        'gt_pop_change_min': _scalar(stats['gt_pop_change_min']),
        'gt_pop_change_25th': _scalar(stats['gt_pop_change_25th']),
        'gt_pop_change_median': _scalar(stats['gt_pop_change_median']),
        'gt_pop_change_75th': _scalar(stats['gt_pop_change_75th']),
        'gt_pop_change_max': _scalar(stats['gt_pop_change_max']),
    }
    if 'n_cell_items' in stats:
        cleaned['n_cell_items'] = int(stats['n_cell_items'])
        cleaned['is_fallback'] = bool(stats['is_fallback'])
    return cleaned


_CLEANERS = {
    'debut': _clean_debut_stats,
    'recent_single_point': _clean_single_point_stats,
    'recent_multi_point': _clean_full_stats,
    'maturing': _clean_full_stats,
}


def save_group_profiles(results, out_dir='.'):
    """
    results: the dict returned by build_group_characteristics, keyed by
    (pop_cluster, pop_direction) -> {'buckets': {...}, 'pooled': {...}}

    Writes:
        {out_dir}/group_profiles.json              -- narration + buckets, LLM-facing + reusable
        {out_dir}/group_mean_embeds.npz             -- backend-only, "{group}__{bucket}" keys
                                                         (now also "{group}__pooled")
        {out_dir}/group_doc_ids.npz                 -- backend-only, reference, same keying
        {out_dir}/group_narrations_for_prompt.json  -- {group: narration}, 8 entries
        {out_dir}/group_narrations_for_prompt.txt   -- same, pre-concatenated
    """
    profiles = {}
    mean_embeds = {}
    doc_ids_by_group = {}

    for key, block in results.items():
        str_key = _group_key_to_str(key)
        pop, direction = key

        buckets_clean = {}
        for bucket, cell in block['buckets'].items():
            buckets_clean[bucket] = _CLEANERS[bucket](cell)
            mean_embeds[f"{str_key}__{bucket}"] = np.asarray(cell['mean_embed'], dtype=np.float32)
            doc_ids_by_group[f"{str_key}__{bucket}"] = np.asarray(cell['doc_ids'])

        pooled = block.get('pooled')
        pooled_clean = _clean_full_stats(pooled) if pooled is not None else None
        if pooled is not None:
            buckets_clean['pooled'] = pooled_clean
            mean_embeds[f"{str_key}__pooled"] = np.asarray(pooled['mean_embed'], dtype=np.float32)
            doc_ids_by_group[f"{str_key}__pooled"] = np.asarray(pooled['doc_ids'])

        group_result_for_narration = {'buckets': block['buckets'], 'pooled': pooled}
        narration = narrate_group((pop, direction), group_result_for_narration)

        profile_entry = {
            'pop_cluster': pop,
            'pop_direction': direction,
            'narration': narration,
            'buckets': buckets_clean,
        }
        if pooled_clean is not None:
            profile_entry['_pooled_fallback_source'] = pooled_clean

        profiles[str_key] = profile_entry

    with open(f"{out_dir}/group_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    np.savez(f"{out_dir}/group_mean_embeds.npz", **mean_embeds)
    np.savez(f"{out_dir}/group_doc_ids.npz", **doc_ids_by_group)

    print(f"Saved {len(profiles)} group profiles to {out_dir}/group_profiles.json, "
          f"{out_dir}/group_mean_embeds.npz ({len(mean_embeds)} vectors), "
          f"{out_dir}/group_doc_ids.npz.")

    save_group_narrations_for_prompt(profiles, out_dir=out_dir)


def load_group_profiles(out_dir='.'):
    with open(f"{out_dir}/group_profiles.json") as f:
        profiles = json.load(f)
    mean_embeds = dict(np.load(f"{out_dir}/group_mean_embeds.npz"))
    doc_ids_by_group = dict(np.load(f"{out_dir}/group_doc_ids.npz"))
    return profiles, mean_embeds, doc_ids_by_group


def load_group_narrations_for_prompt(out_dir='.'):
    with open(f"{out_dir}/group_narrations_for_prompt.json") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Data-prep helper -- builds the columns build_group_characteristics
# needs (pop_cluster, pop_direction, history_depth_bucket, trend_array,
# doc_id) directly from this pipeline's training CSV schema (item_id,
# item_description, release_month, popularity_month, popularity_value,
# timepoint_name, historical_trend, latest_pop_value, latest_momentum).
# Not present in the movie reference, which received these columns
# pre-built from a different upstream pipeline.
# ---------------------------------------------------------------------------

def parse_douban_trend(x):
    if isinstance(x, str):
        return np.array([int(i) for i in x.strip('[]').split()])
    return x

def prepare_dataset_for_groups(df, dataset):
    df = df.reset_index(drop=True).copy()
    # assign doc ID
    df['doc_id'] = np.arange(len(df), dtype=int)

    if dataset == "douban":
        df['trend_array'] = df['trend_array'].apply(parse_douban_trend)
    else:
        df['trend_array'] = df['historical_trend'].apply(parse_historical_trend_to_array)

    df['pop_cluster'] = df['popularity_value'].apply(popularity_threshold)
    df['pop_direction'] = df.apply(
        lambda row: get_retrieved_direction(row['item_age'], row['latest_pop_value'], row['popularity_value']),
        axis=1,
    )
    df['history_depth_bucket'] = df['item_age'].apply(get_history_depth_bucket)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the 8 (pop_cluster, pop_direction) group profiles "
                    "for one dataset (douban, baby or home_and_kitchen)."
    )
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--exp-type', choices=['test','val'], default='test')
    parser.add_argument('--training-data', default=None, help="Override the default data file.")
    parser.add_argument('--embed-npy', default=None, help="Override the default embeddings file.")
    parser.add_argument('--out-dir', default=None, help="Override the default output folder.")
    parser.add_argument('--min-cell-n', type=int, default=DEFAULT_MIN_CELL_N)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    _POP_HIGH = cfg["high"]
    data_path = args.training_data or cfg["data"]
    embed_npy = args.embed_npy or cfg["embed"]
    out_dir = args.out_dir or cfg["out_dir"]
    print(f"Dataset: {args.dataset} | cutoffs: zero=0, medium=1..{_POP_HIGH - 1}, high>={_POP_HIGH}")

    print(f"Loading {data_path} ...")
    if cfg["kind"] == "pickle":
        dataset = pd.read_pickle(data_path)
    else:
        dataset = pd.read_csv(data_path)

    embed = np.load(embed_npy, mmap_mode='r')
    assert len(dataset) == len(embed), (
        f"row-count mismatch: {data_path} has {len(dataset)} rows but "
        f"{embed_npy} has {len(embed)} -- doc_id positions won't line up with embed rows."
    )

    if args.dataset != "douban" and 'item_age' not in dataset.columns:
        def _month_diff(m_from, m_to):
            y1, mo1 = map(int, str(m_from).split('-'))
            y2, mo2 = map(int, str(m_to).split('-'))
            return (y2 - y1) * 12 + (mo2 - mo1)
        dataset['item_age'] = dataset.apply(
            lambda row: _month_diff(row['release_month'], row['popularity_month']) - 1, axis=1
        )

    dataset = prepare_dataset_for_groups(dataset, args.dataset)
    

    print("\npop_cluster counts:");          print(dataset['pop_cluster'].value_counts())
    print("\npop_direction counts:");        print(dataset['pop_direction'].value_counts())
    print("\nhistory_depth_bucket counts:"); print(dataset['history_depth_bucket'].value_counts())

    if args.dataset == 'douban' and args.exp_type == 'val':
        dataset = dataset[dataset['release_date']<'2019-10-01'].copy()
    print(dataset.info())
    print("max date: ", dataset.release_date.max())
        
    results, non_debut_with_trends = build_group_characteristics(
        dataset, embed, min_cell_n=args.min_cell_n
    )
    report_bucket_cell_sizes(results)

    os.makedirs(out_dir, exist_ok=True)
    dataset.to_csv(f"{out_dir}/{args.dataset}_completed_train_data.csv", index=False)
    save_group_profiles(results, out_dir=out_dir)