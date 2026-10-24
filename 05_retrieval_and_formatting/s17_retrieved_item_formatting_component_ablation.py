"""
Retrieved-item formatting for Douban and Amazon (baby / home_and_kitchen): for
each query item's already-computed top_1_center, retrieve the top-K similar
training items and format them as cards (the <SimilarPastItems> context).

The same script covers the main pipeline and its retrieval ablations. Each
axis is a separate flag; run one setting per run and keep the rest at default:

  --group-routing {scored,none}     scored: only the query's top-1 group is searched
                                    none: all groups are searched
  --bucket-match {on,off}           on: candidates must share the query's history-depth
                                    bucket; off: all buckets are allowed
  --item-selection {rerank,embed_only,random,group_top}
                                    rerank: full pipeline (embedding prefilter + re-rank)
                                    embed_only: prefilter + embedding order only
                                    random: random members of the pool
                                    group_top: first K members straight from the group
                                    index (no embedding, no re-rank)
  --ablate-component {none,item_name,trend,relative_change}
                                    card-content ablation, one component dropped per run:
                                    item_name drops the "### <name> (released ...)" header,
                                    trend drops the Trend/history line, relative_change
                                    drops the "Relative change from its own latest known
                                    value: ..." line, none keeps every component
  --e-variants E0,E1                with (E1) / without (E0) explanatory factors
  --g-tag, --ablation-name          labels used only in the output file names

For the random-group ablation, run s14_retrieve_top1_centers.py with --random-group
and pass its output CSV (the *_random_seedN.csv) via --query-top1-csv.

Output file names get an extra suffix (_noItemName / _noTrend / _noRelChange)
when a component is ablated, so they never collide with the un-ablated files.

Run, e.g.:
    python s17_retrieved_item_formatting_component_ablation.py --dataset douban \\
        --ablate-component trend --g-tag G1 --e-variants E1
    python s17_retrieved_item_formatting_component_ablation.py --dataset douban \\
        --group-routing none --g-tag G1 --e-variants E1
"""
import argparse
import os
import re
import pandas as pd
import numpy as np
import json
import html
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "03_clustering_and_indexing"))
from s16_indexing import load_scored_member_index, BUCKET_ORDER
from refined_retrieval_utils import preprocess_desensitized_predictive_signals, strip_valence_tags
from s5_save_eight_groups import get_history_depth_bucket, calculate_normalized_momentum
from dataset_config import DATASETS

# recomputed per category from each category's own real training data --
# (95th percentile of the train-set ground-truth popularity change, instead of
# the raw ground-truth values)
RELATIVE_CHANGE_THRESHOLDS = {
    "home_and_kitchen": {"increase_threshold": 23, "decrease_threshold": -34},
    "baby": {"increase_threshold": 8, "decrease_threshold": -8},
    "douban": {"increase_threshold": 38, "decrease_threshold": -17}
}
# experiment switches
DTW_BACKEND = "tslearn"
GAMMA_POP = 1.0  # decay rate for the recent_single_point latest_pop_value similarity

# re-rank score blend weights
DEFAULT_ALPHA_EMBED = 0.5      # weight on rescaled embedding similarity
DEFAULT_BETA_SECONDARY = 0.5   # weight on the secondary (DTW or pop-value) similarity

if DTW_BACKEND == "tslearn":
    from tslearn.metrics import dtw as _tslearn_dtw

    def _dtw_distance(a, b):
        return float(_tslearn_dtw(a, b))
else:
    raise ValueError(f"unknown DTW_BACKEND: {DTW_BACKEND!r} (expected 'tslearn')")


EMBED_PREFILTER_SIZE = 30

# buckets that use DTW re-rank (step 3, original behavior)
DTW_RERANK_BUCKETS = {"recent_multi_point", "maturing"}
# buckets that use latest_pop_value similarity re-rank instead of DTW
POP_VALUE_RERANK_BUCKETS = {"recent_single_point"}
# buckets that skip re-rank entirely and keep the embedding-similarity order
EMBEDDING_ONLY_BUCKETS = {"debut"}

# Sister-cluster candidates are disabled (links None, 0 slots): not used in the paper.
SISTER_CLUSTER_LINKS = None

SISTER_CLUSTER_SLOTS = 0  # reserved out of EMBED_PREFILTER_SIZE total
DEBUT_PRIOR_ENABLED_CLUSTERS = {}

# ---------------------------------------------------------------------------
# trend_array parsing (float(), not int() -- see s17's module docstring point 2)
# ---------------------------------------------------------------------------

def _parse_trend_array_str(x):
    if not isinstance(x, str):
        return x
    s = x.strip("[]").split()
    return np.array([int(i) for i in s])


# ---------------------------------------------------------------------------
# step 1 -- exact-bucket pool (no backfill)
# ---------------------------------------------------------------------------

def get_exact_bucket_pool(pop_cluster, pop_direction, item_age, scored_index):
    group_key = f"{pop_cluster}__{pop_direction}"
    bucket = get_history_depth_bucket(item_age)
    pool = scored_index.get(group_key, {}).get(bucket, [])
    if not pool:
        print(f"note: empty pool for {group_key}__{bucket} -- no exact-bucket "
              f"candidates for this query.")
    return pool, bucket

def build_candidate_pool(pop_cluster, pop_direction, item_age, scored_index,
                         group_routing="scored", bucket_match=True):
    bucket = get_history_depth_bucket(item_age)
    group_keys = (list(scored_index.keys()) if group_routing == "none"
                  else [f"{pop_cluster}__{pop_direction}"])
    buckets = [bucket] if bucket_match else list(BUCKET_ORDER)

    pool = []
    for g in group_keys:
        cells = scored_index.get(g, {})
        for b in buckets:
            pool.extend(cells.get(b, []))

    seen, deduped = set(), []          # dedupe by doc_id
    for doc_id, score in pool:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append((doc_id, score))
    if not deduped:
        print(f"note: empty candidate pool (group_routing={group_routing}, bucket={bucket}).")
    return deduped, bucket

def get_sister_bucket_pool(pop_cluster, pop_direction, bucket, scored_index):
    if not SISTER_CLUSTER_LINKS:
        return []
    sister_cluster = SISTER_CLUSTER_LINKS.get(pop_cluster)
    if sister_cluster is None:
        return []
    group_key = f"{sister_cluster}__{pop_direction}"
    pool = scored_index.get(group_key, {}).get(bucket, [])
    if not pool:
        print(f"note: no sister-cluster candidates for {group_key}__{bucket} "
              f"(reach from {pop_cluster}) -- proceeding with primary pool only.")
    return pool

# ---------------------------------------------------------------------------
# step 2 -- embedding prefilter
# ---------------------------------------------------------------------------
def check_similarity_distribution(per_query_ranked, valid, train_dataset_by_doc_id,
                                   item_id_col="item_id", high_sim_threshold=0.9):
    all_scores = []
    same_item_high_sim = []
    diff_item_high_sim = []

    for (ranked, bucket), (_, qrow) in zip(per_query_ranked, valid.iterrows()):
        query_item = qrow.get(item_id_col)
        query_doc_id = qrow.get('doc_id')
        for doc_id, embed_score, method, secondary_score, total_score in ranked:
            all_scores.append(embed_score)
            if embed_score >= high_sim_threshold:
                cand_item = train_dataset_by_doc_id.loc[doc_id][item_id_col]
                record = (query_item, query_doc_id, cand_item, doc_id, embed_score)
                if cand_item == query_item:
                    same_item_high_sim.append(record)
                else:
                    diff_item_high_sim.append(record)

    all_scores = np.array(all_scores)
    print(f"embed_score distribution across {len(all_scores)} retrieved (query, candidate) pairs:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(all_scores, p):.4f}")
    print(f"  mean: {all_scores.mean():.4f}, max: {all_scores.max():.4f}")

    n_high = len(same_item_high_sim) + len(diff_item_high_sim)
    print(f"\npairs with embed_score >= {high_sim_threshold}: {n_high}")
    print(f"  same-item, different timepoint (legitimate, don't exclude): {len(same_item_high_sim)}")
    print(f"  different-item near-duplicates (candidates for exclusion): {len(diff_item_high_sim)}")

    if diff_item_high_sim:
        print(f"\nsample different-item high-similarity pairs:")
        for query_item, query_doc_id, cand_item, cand_doc_id, score in diff_item_high_sim[:10]:
            print(f"  item_id={query_item!r} (doc_id={query_doc_id}) <-> "
                  f"item_id={cand_item!r} (doc_id={cand_doc_id}): {score:.4f}")

    return all_scores, same_item_high_sim, diff_item_high_sim

def cosine_similarity_matrix(query_vec, candidate_matrix):
    query_vec = np.asarray(query_vec, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float64)

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    cand_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-12
    cand_unit = candidate_matrix / cand_norms
    return cand_unit @ query_norm


def rescale_cosine_to_unit(cosine_sim):
    return (float(cosine_sim) + 1.0) / 2.0


def embedding_prefilter(pool_doc_ids, query_embedding, train_embeddings,
                         doc_id_to_embed_idx, prefilter_size=EMBED_PREFILTER_SIZE):
    valid_doc_ids = [d for d in pool_doc_ids if d in doc_id_to_embed_idx]
    n_missing = len(pool_doc_ids) - len(valid_doc_ids)
    if n_missing:
        print(f"note: {n_missing}/{len(pool_doc_ids)} pool candidates have no "
              f"embedding row -- skipping them.")
    if not valid_doc_ids:
        return []

    candidate_rows = np.array([doc_id_to_embed_idx[d] for d in valid_doc_ids])
    candidate_matrix = train_embeddings[candidate_rows]
    sims = cosine_similarity_matrix(query_embedding, candidate_matrix)

    order = np.argsort(-sims)[:prefilter_size]
    return [(valid_doc_ids[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# step 3a -- DTW re-rank (recent_multi_point / maturing)
# ---------------------------------------------------------------------------
def parse_center_name(center_name):
    if not isinstance(center_name, str) or "__" not in center_name:
        return None, None
    parts = center_name.split("__")
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]

def _as_dtw_series(trend_array):
    arr = trend_array.tolist() if hasattr(trend_array, "tolist") else list(trend_array)
    return np.array(arr, dtype=np.float64)


def _is_unusable_trend(trend_list):
    if len(trend_list) == 0:
        return True
    if len(trend_list) == 1 and trend_list[0] == -1:
        return True
    arr = np.asarray(trend_list, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return True
    return False


def dtw_distance_to_similarity(distance):
    return 1.0 / (1.0 + distance)


def dtw_rerank(query_trend_array, shortlist, dataset_by_doc_id, trend_col="trend_array",
               alpha_embed=DEFAULT_ALPHA_EMBED, beta_secondary=DEFAULT_BETA_SECONDARY):
    if _is_unusable_trend(query_trend_array):
        print("note: query item's own trend_array is unusable (sentinel, zero-length, "
            "or all-NaN) -- cannot DTW-rerank this query; falling back to "
            "embedding-similarity order instead.")
        fallback = [(doc_id, embed_score, "dtw_unavailable", None, rescale_cosine_to_unit(embed_score))
                    for doc_id, embed_score in shortlist]
        fallback.sort(key=lambda t: t[4], reverse=True)
        return fallback

    query_series = _as_dtw_series(query_trend_array)

    scored = []
    excluded_doc_ids = []
    for doc_id, embed_score in shortlist:
        cand_trend = dataset_by_doc_id.loc[doc_id][trend_col]
        if _is_unusable_trend(cand_trend):
            excluded_doc_ids.append(doc_id)
            continue
        cand_series = _as_dtw_series(cand_trend)
        distance = _dtw_distance(query_series, cand_series)
        similarity = dtw_distance_to_similarity(distance)
        embed_score_rescaled = rescale_cosine_to_unit(embed_score)
        total_score = alpha_embed * embed_score_rescaled + beta_secondary * similarity
        scored.append((doc_id, embed_score, "dtw", similarity, total_score))

    if excluded_doc_ids:
        print(f"note: excluded {len(excluded_doc_ids)}/{len(shortlist)} candidates from "
              f"DTW re-rank -- unusable trend_array (sentinel, zero-length, or all-NaN): "
              f"{excluded_doc_ids}")

    scored.sort(key=lambda t: t[4], reverse=True)  # descending total score
    return scored


# ---------------------------------------------------------------------------
# step 3b -- latest_pop_value similarity re-rank (recent_single_point)
# ---------------------------------------------------------------------------

def pop_normalization(inf_value, center_value, gamma=1.0):
    return float(np.exp(-abs(float(center_value) - float(inf_value)) / gamma))


def pop_value_rerank(query_latest_pop_value, shortlist, dataset_by_doc_id,
                      latest_pop_value_col="latest_pop_value", gamma_pop=1.0,
                      alpha_embed=DEFAULT_ALPHA_EMBED, beta_secondary=DEFAULT_BETA_SECONDARY):
    scored = []
    n_no_latest = 0
    for doc_id, embed_score in shortlist:
        cand_latest = dataset_by_doc_id.loc[doc_id][latest_pop_value_col]
        if pd.isna(cand_latest):          # e.g. debut candidates when bucket_match is off; same rule as DTW
            n_no_latest += 1
            continue
        similarity = pop_normalization(query_latest_pop_value, cand_latest, gamma=gamma_pop)
        embed_score_rescaled = rescale_cosine_to_unit(embed_score)
        total_score = alpha_embed * embed_score_rescaled + beta_secondary * similarity
        scored.append((doc_id, embed_score, "pop_value_sim", similarity, total_score))

    if n_no_latest:
        print(f"note: excluded {n_no_latest}/{len(shortlist)} candidates from pop-value re-rank -- no latest_pop_value.")

    scored.sort(key=lambda t: t[4], reverse=True)  # descending total score
    return scored


# ---------------------------------------------------------------------------
# step 3c -- embedding-only (debut): no re-rank, keep embedding order
# ---------------------------------------------------------------------------

def embedding_only_rerank(shortlist):
    reshaped = [(doc_id, embed_score, "embedding_only", None, rescale_cosine_to_unit(embed_score))
                for doc_id, embed_score in shortlist]
    reshaped.sort(key=lambda t: t[4], reverse=True)
    return reshaped

# ---------------------------------------------------------------------------
# step 4 -- top K, end to end
# ---------------------------------------------------------------------------

def retrieve_bucket_embed_rerank(pop_cluster, pop_direction, item_age, k, scored_index,
                                  query_embedding, query_trend_array, query_latest_pop_value,
                                  train_embeddings, doc_id_to_embed_idx,
                                  dataset_by_doc_id, trend_col="trend_array",
                                  latest_pop_value_col="latest_pop_value",
                                  prefilter_size=EMBED_PREFILTER_SIZE, gamma_pop=1.0,
                                  alpha_embed=DEFAULT_ALPHA_EMBED, beta_secondary=DEFAULT_BETA_SECONDARY,
                                  group_routing="scored", bucket_match=True, item_selection="rerank", rng=None):
    pool, bucket = build_candidate_pool(pop_cluster, pop_direction, item_age, scored_index,
                                        group_routing=group_routing, bucket_match=bucket_match)
    pool_doc_ids = [doc_id for doc_id, _score in pool]

    sister_doc_ids = ([doc_id for doc_id, _s in
                       get_sister_bucket_pool(pop_cluster, pop_direction, bucket, scored_index)]
                      if group_routing == "scored" else [])

    if not pool_doc_ids and not sister_doc_ids:
        return [], bucket

    if item_selection == "random":
        n = min(k, len(pool_doc_ids))
        chosen = rng.choice(np.array(pool_doc_ids), size=n, replace=False)
        return [(int(d), None, "random", None, None) for d in chosen], bucket

    if item_selection == "group_top":
        return [(int(d), None, "group_top", sc, None) for d, sc in pool[:k]], bucket

    if sister_doc_ids:
        shortlist = (
            embedding_prefilter(pool_doc_ids, query_embedding, train_embeddings,
                                 doc_id_to_embed_idx, prefilter_size=prefilter_size - SISTER_CLUSTER_SLOTS)
            + embedding_prefilter(sister_doc_ids, query_embedding, train_embeddings,
                                   doc_id_to_embed_idx, prefilter_size=SISTER_CLUSTER_SLOTS)
        )
    else:
        shortlist = embedding_prefilter(pool_doc_ids, query_embedding, train_embeddings,
                                         doc_id_to_embed_idx, prefilter_size=prefilter_size)

    if not shortlist:
        return [], bucket

    if item_selection == "embed_only":
        ranked = embedding_only_rerank(shortlist)
    elif bucket in DTW_RERANK_BUCKETS:
        ranked = dtw_rerank(query_trend_array, shortlist, dataset_by_doc_id, trend_col=trend_col,
                             alpha_embed=alpha_embed, beta_secondary=beta_secondary)
    elif bucket in POP_VALUE_RERANK_BUCKETS:
        ranked = pop_value_rerank(
            query_latest_pop_value, shortlist, dataset_by_doc_id,
            latest_pop_value_col=latest_pop_value_col, gamma_pop=gamma_pop,
            alpha_embed=alpha_embed, beta_secondary=beta_secondary,
        )
    elif bucket in EMBEDDING_ONLY_BUCKETS:
        ranked = embedding_only_rerank(shortlist)
    else:
        raise ValueError(
            f"bucket {bucket!r} (from history_depth_bucket) isn't mapped to any "
            f"re-rank method -- check DTW_RERANK_BUCKETS / POP_VALUE_RERANK_BUCKETS / "
            f"EMBEDDING_ONLY_BUCKETS against BUCKET_ORDER={BUCKET_ORDER}."
        )

    if len(ranked) < k:
        print(f"note: only {len(ranked)}/{k} candidates survived re-ranking for "
              f"{pop_cluster}__{pop_direction}__{bucket} (some may have been excluded "
              f"for an unusable trend_array) -- that's the ceiling here, not a bug.")

    return ranked[:k], bucket

# ---------------------------------------------------------------------------
# per-item formatting helpers
# ---------------------------------------------------------------------------

def _month_label(date_value):
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M"))
    except (ValueError, TypeError):
        return None


def _month_before(date_value):
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M") - 1)
    except (ValueError, TypeError):
        return None


def _format_trend_for_card(trend_array, release_date=None, popularity_month=None):
    trend_list = trend_array.tolist() if hasattr(trend_array, "tolist") else list(trend_array)
    if len(trend_list) == 0 or (len(trend_list) == 1 and trend_list[0] == -1):
        return "Trend", "No historical trend"

    start_label = _month_label(release_date) if release_date is not None else None
    end_label = _month_before(popularity_month) if popularity_month is not None else None

    if start_label and end_label:
        window = f", {start_label}" if start_label == end_label else f", {start_label} to {end_label}"
        window += " (monthly)"
    else:
        window = ""
    if len(trend_list) > 10:
        truncated = f"{', '.join(map(str, trend_list[:5]))}, ..., {', '.join(map(str, trend_list[-5:]))}"
        label = f"Trend{window} (first and last 5 shown)" if window else "Trend (first and last 5 months)"
        return label, truncated
    label = f"Trend{window}" if window else "Trend (full history)"
    return label, ", ".join(map(str, trend_list))


def _format_relative_change_for_case(row, pop_value_col, latest_pop_value_col,
                                      increase_threshold, decrease_threshold):
    if pop_value_col not in row or latest_pop_value_col not in row:
        return None
    outcome, latest = row[pop_value_col], row[latest_pop_value_col]
    if pd.isna(outcome) or pd.isna(latest):
        return None
    outcome, latest = float(outcome), float(latest)
    if abs(latest) < 1e-6:
        return None  # no meaningful baseline -- fall back to tier label

    raw_change = outcome - latest
    if not (raw_change >= increase_threshold or raw_change <= decrease_threshold):
        return None  # below the outstanding-change bar -- don't surface it

    pct_change = raw_change / abs(latest) * 100.0
    pct_change = max(min(pct_change, 500.0), -100.0)  # clip; -100% floor, popularity can't go negative
    return f"{'+' if pct_change > 0 else ''}{pct_change:.0f}%"


def format_debut_outcome_prior(pop_cluster, pop_direction, bucket, profile_stats, small_n_threshold=30):
    if pop_cluster not in DEBUT_PRIOR_ENABLED_CLUSTERS:
        return None

    if "outcome_value_25th" not in profile_stats or "outcome_value_75th" not in profile_stats:
        return None  # profile wasn't built with debut outcome-value stats

    usual_low = profile_stats["outcome_value_25th"]
    usual_high = profile_stats["outcome_value_75th"]
    true_min = profile_stats.get("outcome_value_min")
    true_max = profile_stats.get("outcome_value_max")

    small_n_note = (
        f" (small sample, n={profile_stats.get('n_items', 0)} -- treat as a rough "
        f"indication only)"
        if profile_stats.get("n_items", 0) < small_n_threshold else ""
    )

    extreme_parts = []
    if true_min is not None and true_min < usual_low:
        extreme_parts.append(f"as low as {true_min:.0f}")
    if true_max is not None and true_max > usual_high:
        extreme_parts.append(f"as high as {true_max:.0f}")
    extremes_note = (
        f" with rare items {' or '.join(extreme_parts)} "
        f"-- reserve those extremes only for items whose own factors clearly justify them."
        if extreme_parts else ""
    )

    return (
            f"Outcome-popularity context for {pop_cluster} / {pop_direction} ({bucket}): "
            f"most items in this exact group land between {usual_low:.0f} and {usual_high:.0f}"
            f"{small_n_note}.{extremes_note} This describes the retrieved bucket, not a claim that the target belongs in it -- retrieval on tier/magnitude is less reliable than on direction."
            f"Treat this only as a sanity check against implausible outputs, not as a target range to converge toward."
    )


_PRODUCT_NAME_RE = re.compile(r'^The\s+\w+\s+product name is\s+(.*?)\s*;', re.IGNORECASE)


def _extract_product_name(description, max_len=80):
    if not isinstance(description, str) or not description.strip():
        return "(no description)"
    decoded = html.unescape(description)
    match = _PRODUCT_NAME_RE.match(decoded)
    name = match.group(1).strip() if match else decoded.strip()
    if not name:
        return "(no product name)"
    return name[:max_len] + ("..." if len(name) > max_len else "")


def format_retrieved_context_minimal(ranked_members, dataset_by_doc_id, top_1_center,
                                      item_id_col="item_id", release_date_col="release_month",
                                      description_col="item_description",
                                      trend_col="trend_array", predictive_signals_col="explanatory_factors",
                                      pop_value_col="popularity_value", latest_pop_value_col="latest_pop_value",
                                      increase_threshold=38, decrease_threshold=-17,
                                      include_explanatory_factors=True,
                                      ablate_component="none"):
    """Same 5-field card as the un-ablated format, plus one pass-through:
    ablate_component in {'none', 'item_name', 'trend', 'relative_change'}
    drops exactly that one component from every card in this call (see
    module docstring)."""
    blocks = []
    for doc_id, embed_score, method, secondary_score, total_score in ranked_members:
        row = dataset_by_doc_id.loc[doc_id]
        blocks.append(_format_one_case_minimal(
            row, item_id_col, release_date_col, description_col, trend_col,
            predictive_signals_col, pop_value_col, latest_pop_value_col,
            increase_threshold, decrease_threshold,
            include_explanatory_factors=include_explanatory_factors,
            ablate_component=ablate_component,
        ))
    return "\n\n".join(blocks)

def _format_one_case_minimal(row, item_id_col, release_date_col, description_col,
                              trend_col, predictive_signals_col,
                              pop_value_col, latest_pop_value_col,
                              increase_threshold, decrease_threshold,
                              popularity_month_col="popularity_month",
                              include_explanatory_factors=True,
                              ablate_component="none"):
    """Identical to the un-ablated one-case formatter,
    except each of the three ablatable lines is only appended when
    ablate_component doesn't name it:
      - 'item_name' drops the whole "### <name> (released <date>)" header
        line -- for Douban the item name and its release date are rendered
        on that single line together, so ablating "item name" necessarily
        drops the release date with it too (there is no header without a
        name to attach the date to).
      - 'trend' drops the Trend/history line.
      - 'relative_change' drops the "Relative change from its own latest
        known value: ..." line.
    Exactly one of these is dropped per run (ablate_component is a single
    choice, not a set) -- see module docstring for why."""
    lines = []

    if ablate_component != "item_name":
        if item_id_col == 'item_id':
            product_name = _extract_product_name(row.get(description_col))
            header = f"### {product_name} released {row[release_date_col]}"      # Amazon (unchanged)
        else:
            product_name = row['movie_name']
            header = f"### {product_name} (released {row[release_date_col]})"    # Douban: matches main-table contexts
        lines.append(header)

    if ablate_component != "relative_change":
        pct_change = _format_relative_change_for_case(
            row, pop_value_col, latest_pop_value_col, increase_threshold, decrease_threshold
        )
        if pct_change is not None:
            # non-debut row: show the quantitative change only, no direction label
            lines.append(f"Relative change from its own latest known value: {pct_change}")
        # else: pct_change is None -> latest_pop_value is absent (debut row) or
        # the change didn't clear the outstanding-change bar.

    if ablate_component != "trend":
        trend_label, trend_str = _format_trend_for_card(
            row[trend_col],
            release_date=row.get(release_date_col),
            popularity_month=row.get(popularity_month_col),
        )
        if trend_str != "No historical trend":
            lines.append(f"{trend_label}: {trend_str}")

    if include_explanatory_factors:
        signals_raw = row.get(predictive_signals_col)
        signals_clean = (
        strip_valence_tags(preprocess_desensitized_predictive_signals(signals_raw))
        if pd.notna(signals_raw) and signals_raw != ""
        else "(no notes for this case)"
    )
        lines.append(f"Explanatory factors:\n{signals_clean}")

    return "\n".join(lines)

def format_group_prior(pop_cluster, pop_direction, bucket, profile_stats, small_n_threshold=30):
    if bucket != "debut":
        return None
    return format_debut_outcome_prior(pop_cluster, pop_direction, bucket, profile_stats, small_n_threshold)

def report_index_order(scored_index):
    n_dec = n_inc = n_other = n_const = 0
    for cells in scored_index.values():
        for pool in cells.values():
            sc = [s for _, s in pool]
            if len(sc) < 2:
                continue
            dec = all(a >= b for a, b in zip(sc, sc[1:]))
            inc = all(a <= b for a, b in zip(sc, sc[1:]))
            if dec and inc:
                n_const += 1
            elif dec:
                n_dec += 1
            elif inc:
                n_inc += 1
            else:
                n_other += 1
    print(f"[group_top] index cell order: non-increasing={n_dec}, non-decreasing={n_inc}, "
          f"constant={n_const}, unsorted={n_other}")
    if n_inc or n_other:
        print("[group_top] WARNING: not all cells are stored best-first -- group_top would NOT be the top-scored members.")


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval context (top-K similar training items, "
                     "formatted as cards) for each query item's already-"
                     "computed top_1_center, for one Amazon/Douban category -- "
                     "with one retrieved-card COMPONENT ablated at a time "
                     "(--ablate-component)."
    )
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--mode', choices=['active', 'inactive'], default='active',
                         help="Matches s14_retrieve_top1_centers.py's mode suffix.")
    parser.add_argument('--train-data-path', default=None,
                         help="Defaults to {train_out_dir}/{name}_completed_train_data.csv")
    parser.add_argument('--query-top1-csv', default=None,
                         help="Defaults to processed_results/{name}/{name}_top_1_centers_{mode}.csv "
                              "(for random-group ablation pass the s14 *_random_seedN.csv here)")
    parser.add_argument('--query-features-csv', default=None,
                         help="Defaults to processed_results/{name}/{name}_feature_selection_merged.csv")
    parser.add_argument('--query-embed-path', default=None,
                         help="Defaults to {name}_test_embed/final_embeddings.npy")
    parser.add_argument('--output-dir', default=None,
                         help="Defaults to processed_results/{name}/retrieved_context")
    parser.add_argument('--k-values', default="1,5,10",
                         help="Comma-separated K values, e.g. '1,5,10'.")
    parser.add_argument('--alpha-embed', type=float, default=DEFAULT_ALPHA_EMBED)
    parser.add_argument('--beta-secondary', type=float, default=DEFAULT_BETA_SECONDARY)
    parser.add_argument('--gamma-pop', type=float, default=GAMMA_POP)
    parser.add_argument('--include-gt-pop-change-prior', action='store_true', default=False)
    parser.add_argument('--group-routing', choices=['scored', 'none'], default='scored')
    parser.add_argument('--bucket-match', choices=['on', 'off'], default='on')
    parser.add_argument('--item-selection', choices=['rerank', 'embed_only', 'random', 'group_top'], default='rerank',
                        help="rerank = full pipeline | embed_only = prefilter + embedding order | random | "
                             "group_top = first K members straight from the group index (no embedding, no rerank)")
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--e-variants', default='E1',
                        help="Comma-separated: E0 (no explanatory factors), E1 (with).")
    parser.add_argument('--g-tag', default=None, help="G0/G1 label, filenames only.")
    parser.add_argument('--ablation-name', default=None, help="Label for filenames only.")
    parser.add_argument('--ablate-component', choices=['none', 'item_name', 'trend', 'relative_change'],
                        default='none',
                        help="Drop exactly one retrieved-card component for this run: "
                             "'item_name' drops the '### <name> (released <date>)' header line, "
                             "'trend' drops the Trend/history line, "
                             "'relative_change' drops the 'Relative change from its own latest "
                             "known value: ...' line. 'none' (default) keeps all components -- "
                             "same card content as an un-ablated run.")
    args = parser.parse_args()

    name = args.dataset
    cfg = DATASETS[args.dataset]
    train_out_dir = cfg["out_dir"]
    train_data_path = args.train_data_path or f"{train_out_dir}/{name}_completed_train_data.csv"
    train_embed_path = cfg["embed"]
    output_dir = args.output_dir or f"processed_results/{name}/retrieved_context"
    query_top1_csv = args.query_top1_csv or f"processed_results/{name}/{name}_top_1_centers_{args.mode}.csv"
    query_features_csv = args.query_features_csv or f"processed_results/{name}/{name}_feature_selection_merged.csv"
    query_embed_path = args.query_embed_path or f"{name}_test_embed/final_embeddings.npy"
    k_values = [int(k) for k in args.k_values.split(",")]
    e_variants = [e.strip() for e in args.e_variants.split(",")]
    bucket_match = (args.bucket_match == "on")
    thresholds = RELATIVE_CHANGE_THRESHOLDS[name]

    # ---- load training-side artifacts ----
    train_dataset = pd.read_csv(train_data_path)
    train_dataset["trend_array"] = train_dataset["trend_array"].apply(_parse_trend_array_str)
    print(f"original retrieved dataset: {train_dataset.shape}")
    if "latest_momentum" not in train_dataset.columns:
        train_dataset["latest_momentum"] = train_dataset['trend_array'].apply(calculate_normalized_momentum)
    train_dataset_by_doc_id = train_dataset.set_index("doc_id", drop=False)

    scored_index = load_scored_member_index(f"{train_out_dir}/{name}_group_scored_members.json")
    with open(f"{train_out_dir}/group_profiles.json") as f:
        profiles = json.load(f)
    if args.item_selection == "group_top":
        report_index_order(scored_index)

    # ---- load embeddings ----
    train_embeddings = np.load(train_embed_path)
    query_embeddings = np.load(query_embed_path)
    train_doc_id_to_embed_idx = {
        int(doc_id): int(doc_id) for doc_id in train_dataset["doc_id"]
    }

    # ---- load query-side results ----
    query_features = pd.read_csv(query_features_csv)
    query_features["trend_array"] = query_features["trend_array"].apply(_parse_trend_array_str)
    if "doc_id" not in query_features.columns:
        query_features["doc_id"] = list(range(len(query_features)))

    # assert len(query_features) == query_embeddings.shape[0], (
    #     f"{query_features_csv} has {len(query_features)} rows but {query_embed_path} has "
    #     f"{query_embeddings.shape[0]} -- row-order alignment assumption is broken."
    # )
    query_doc_id_to_embed_idx = {
        int(doc_id): int(doc_id) for doc_id in query_features["doc_id"]
    }

    if args.group_routing == "none":
        # no outcome-group routing: top-1 file is not needed at all
        query_df = query_features.copy()
        query_df["retrieved_top_1_center"] = "all__all"      # placeholder, unused
    else:
        query_top1 = pd.read_csv(query_top1_csv)
        query_top1_subset = query_top1[['doc_id', 'top_1_center']].copy()
        query_df = query_features.merge(query_top1_subset, on=["doc_id"], how="left")
        parsed = query_df["top_1_center"].apply(parse_center_name)
        query_df["retrieved_pop_cluster"] = parsed.apply(lambda t: t[0])
        query_df["retrieved_direction"] = parsed.apply(lambda t: t[1])
        query_df['retrieved_top_1_center'] = (
            query_df["retrieved_pop_cluster"] + "__" + query_df["retrieved_direction"]
        )

    valid = query_df[query_df["retrieved_top_1_center"].notna() & query_df["item_age"].notna()].copy()
    n_dropped = len(query_df) - len(valid)
    if n_dropped:
        print(f"Excluding {n_dropped} query rows with no top_1_center (scoring "
              f"error) or no item_age match from {query_features_csv}.")
    print(f"Building retrieval context for {len(valid)} query items "
          f"(group_routing={args.group_routing}, bucket_match={args.bucket_match}, "
          f"item_selection={args.item_selection}, ablate_component={args.ablate_component})...")
    dropped = query_df.drop(valid.index)
    if len(dropped):
        print(f"Excluding {len(dropped)} query rows:")
        print(dropped[[c for c in ["doc_id", "movie_id", "movie_name", "item_age", "retrieved_top_1_center"]
                    if c in dropped.columns]].to_string(index=False))

    # ---- run once per query item at max_k; slice per K later ----
    max_k = max(k_values)
    per_query_ranked = []
    for i, (_, qrow) in enumerate(valid.iterrows()):
        if i % 500 == 0:
            print(f"  {i}/{len(valid)}...")
        try:
            embed_idx = query_doc_id_to_embed_idx.get(qrow["doc_id"])
            if embed_idx is None:
                raise ValueError(f"doc_id={qrow['doc_id']} not found in query embedding index")
            query_embedding = query_embeddings[embed_idx]
            pop_cluster, pop_direction = qrow["retrieved_top_1_center"].split("__", 1)
            item_rng = np.random.default_rng([args.random_seed, int(qrow["doc_id"])])
            ranked, bucket = retrieve_bucket_embed_rerank(
                pop_cluster, pop_direction, qrow["item_age"], max_k, scored_index,
                query_embedding=query_embedding,
                query_trend_array=qrow["trend_array"],
                query_latest_pop_value=qrow["latest_pop_value"],
                train_embeddings=train_embeddings,
                doc_id_to_embed_idx=train_doc_id_to_embed_idx,
                dataset_by_doc_id=train_dataset_by_doc_id,
                gamma_pop=args.gamma_pop,
                alpha_embed=args.alpha_embed,
                beta_secondary=args.beta_secondary,
                group_routing=args.group_routing,
                bucket_match=bucket_match,
                item_selection=args.item_selection,
                rng=item_rng,
            )
        except Exception as e:
            print(f"  error building context for doc_id={qrow['doc_id']}: {e}")
            ranked, bucket = [], None
        per_query_ranked.append((ranked, bucket))

    if args.item_selection not in ("random", "group_top") and args.dataset == 'douban':   # embed_score is None in these modes
        check_similarity_distribution(per_query_ranked, valid, train_dataset_by_doc_id,item_id_col="movie_id")
    elif args.item_selection not in ("random", "group_top"):
        check_similarity_distribution(per_query_ranked, valid, train_dataset_by_doc_id)

    invalid_columns = ['custom_id', "is_valid",
                        "embed_sim__high_popular__decrease", "embed_sim__high_popular__increase",
                        "embed_sim__high_popular__stable", "embed_sim__medium_popular__decrease",
                        "embed_sim__medium_popular__increase", "embed_sim__medium_popular__stable",
                        "embed_sim__zero_popular__decrease", "embed_sim__zero_popular__stable", "reasoning"]
    keep_cols = [c for c in valid.columns if c not in invalid_columns]

    def build_contexts(k, include_explanatory):
        contexts = []
        for (ranked, bucket), (_, qrow) in zip(per_query_ranked, valid.iterrows()):
            if not ranked:
                contexts.append("")
                continue
            top_k_ranked = ranked[:k]
            if k > 1:
                top_k_ranked = top_k_ranked[::-1]
            if name == 'douban':
                cards_text = format_retrieved_context_minimal(
                    top_k_ranked, train_dataset_by_doc_id, qrow["retrieved_top_1_center"],
                    item_id_col="movie_id", release_date_col="release_date", description_col='narration',
                    trend_col="trend_array", predictive_signals_col="explanatory_factors",
                    pop_value_col="popularity_value", latest_pop_value_col="latest_pop_value",
                    increase_threshold=thresholds["increase_threshold"],
                    decrease_threshold=thresholds["decrease_threshold"],
                    include_explanatory_factors=include_explanatory,
                    ablate_component=args.ablate_component,
                )
            else:
                cards_text = format_retrieved_context_minimal(
                    top_k_ranked, train_dataset_by_doc_id, qrow["retrieved_top_1_center"],
                    increase_threshold=thresholds["increase_threshold"],
                    decrease_threshold=thresholds["decrease_threshold"],
                    include_explanatory_factors=include_explanatory,
                    ablate_component=args.ablate_component,
                )

            context_text = cards_text
            if args.include_gt_pop_change_prior:
                pop_cluster, pop_direction = qrow["retrieved_top_1_center"].split("__", 1)
                group_key = qrow["retrieved_top_1_center"]
                bucket_stats = profiles.get(group_key, {}).get("buckets", {}).get(bucket) if bucket else None
                if bucket_stats is None:
                    print(f"note: no profile found for {group_key}__{bucket}, skipping gt_pop_change prior.")
                else:
                    prior_line = format_group_prior(pop_cluster, pop_direction, bucket, bucket_stats)
                    if prior_line is not None:
                        context_text = f"{prior_line}\n\n{cards_text}"
            contexts.append(context_text)
        return contexts

    # filenames: an extra tag when a component is ablated, so these never
    # collide with the un-ablated files.
    ABLATE_TAG = {
        "none": "",
        "item_name": "_noItemName",
        "trend": "_noTrend",
        "relative_change": "_noRelChange",
    }[args.ablate_component]

    os.makedirs(output_dir, exist_ok=True)
    for k in k_values:
        for ev in e_variants:
            contexts = build_contexts(k, include_explanatory=(ev == "E1"))
            out_df = valid[keep_cols].copy()
            out_df["retrieved_context"] = contexts
            out_df["retrieved_ids"] = [json.dumps([int(m[0]) for m in ranked[:k]]) for ranked, _ in per_query_ranked]
            out_df["retrieved_bucket"] = [b for _, b in per_query_ranked]

            variant_tag = (
                f"embed_rerank_k{k}"
                f"_a{int(round(args.alpha_embed * 100))}_b{int(round(args.beta_secondary * 100))}"
                f"_gtprior{int(args.include_gt_pop_change_prior)}"
            )
            parts = []
            if args.ablation_name:
                parts.append(args.ablation_name)
            if args.g_tag:
                parts.append(f"{args.g_tag}{ev}")
            elif ev == "E0" or len(e_variants) > 1:
                parts.append(ev)
            suffix = ("_" + "_".join(parts)) if parts else ""
            suffix += ABLATE_TAG

            output_csv = f"{output_dir}/{name}_retrieval_context_per_query_{variant_tag}{suffix}.csv"
            out_df.to_csv(output_csv, index=False)
            print(f"Saved retrieval context (K={k}, {ev}, ablate_component={args.ablate_component}) "
                  f"for {len(out_df)} query items -> {output_csv}")


if __name__ == "__main__":
    main()