import json
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# candidate feature -> group_profiles.json median-key mapping
# ---------------------------------------------------------------------------
NONNEGATIVE_LOG_FEATURES = {
    'item_age','item_age_days', 'latest_pop_value', 'trend_min', 'trend_max',
    'trend_volatility', 'time_since_peak',
}
SIGNED_LOG_FEATURES = {'trend_overall_change'}
# deliberately untransformed: 'embed_similarity' (cosine-based),
# 'latest_momentum' (already an average log1p-diff internally)

FEATURE_TO_CENTER_KEY = {
    "latest_pop_value": "latest_pop_value_median",
    "item_age": "item_age_median",
    "item_age_days": "item_age_days_median",
    "trend_min": "trend_min_median",
    "trend_max": "trend_max_median",
    "trend_volatility": "trend_volatility_median",
    "trend_overall_change": "trend_overall_change_median",
    "time_since_peak": "time_since_peak_median_multi_point",
    "latest_momentum": "latest_mom_median",
}

# rank-position weights, keyed by how many features were selected
RANK_WEIGHTS = {
    1: [1.0],              # debut's only ever-selected feature: embed_similarity
    2: [0.70, 0.30],
    3: [0.50, 0.30, 0.20],
}

# RANK_WEIGHTS = {
#     1: [1.0],              # debut's only ever-selected feature: embed_similarity
#     2: [0.30, 0.70],
#     3: [0.20, 0.30, 0.50],
# }

# xtreme weights (best overall)
# RANK_WEIGHTS = {
#     1: [1.0],              # debut's only ever-selected feature: embed_similarity
#     2: [0.85, 0.15],
#     3: [0.40, 0.35, 0.20],
# }

# reproducible/comparable across runs
ALL_CANDIDATE_FEATURES = [
    "latest_pop_value", "item_age", "trend_min", "trend_max",
    "trend_volatility", "trend_overall_change", "time_since_peak",
    "latest_momentum", "embed_similarity",
]

# the four bucket names s5_save_eight_groups.py's get_history_depth_bucket
# can produce -- used below to parse "{group}__{bucket}" keys back apart
BUCKET_NAMES = ("debut", "recent_single_point", "recent_multi_point", "maturing")


TIER_PHRASE = {
    "zero_popular": "minimal popularity",
    "medium_popular": "moderate popularity",
    "high_popular": "high popularity",
}

# Targets the specific, recurring "(a/an) (popularity) score {of|was|in
# <month>} <N>" citation pattern -- NOT every number in the text. Trend
# narration like "fell from 148 ... to 20" is deliberately left alone,
# since that's magnitude-matched by the DTW/pop-value re-rank and follows
# the same "keep trend numbers raw" rule already applied to trend_array.
_SCORE_MENTION_RE = re.compile(
    r'(?:an?\s+)?(?:popularity\s+)?score\s+'
    r'(?:of|was|in\s+\d{4}-\d{2}(?:,|\s+of|\s+was)?)\s*(\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Cleaning functions for numeric mentions in explanatory text
# ---------------------------------------------------------------------------
def desensitize_score_mentions(text):
    if not isinstance(text, str) or not text:
        return text
    def _replace(m):
        return TIER_PHRASE[popularity_threshold(float(m.group(1)))]
    return _SCORE_MENTION_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# center loading
# ---------------------------------------------------------------------------

def load_center_profiles(profiles_path="group_profiles.json", mean_embeds_path="group_mean_embeds.npz"):
    """Returns dict: "pop_cluster__direction__bucket" -> {..bucket's cleaned
    stats.., 'embed_vector': np.ndarray}. A group/bucket combination only
    appears here if that cell actually had >=1 real training item --
    s5_save_eight_groups.py never writes a key for an empty cell. debut
    cells carry only n_items/n_cell_items/is_fallback plus 'embed_vector'
    (no scalar stat keys at all -- FEATURE_TO_CENTER_KEY lookups would
    KeyError there, which is correct: embed_similarity is the only feature
    ever scored against a debut center, enforced in
    s14_retrieve_top1_centers.py before composite_score is ever called for
    a debut item)."""
    with open(profiles_path) as f:
        profiles = json.load(f)

    mean_embeds = dict(np.load(mean_embeds_path)) if mean_embeds_path else {}

    centers = {}
    for group_key, profile in profiles.items():
        for bucket, bucket_stats in profile.get("buckets", {}).items():
            full_key = f"{group_key}__{bucket}"
            center = dict(bucket_stats)
            if full_key in mean_embeds:
                center["embed_vector"] = mean_embeds[full_key]
            centers[full_key] = center
    return centers


def group_keys_from_centers(centers):
    """Recovers the flat list of "pop__direction" group names from a
    centers dict keyed "pop__direction__bucket" -- i.e. the 8 candidate
    GROUPS, independent of bucket. Used to build this item's candidate set
    for a specific bucket (see candidate_keys_for_bucket) without needing a
    second file read."""
    groups = set()
    for key in centers:
        for bucket in BUCKET_NAMES:
            suffix = f"__{bucket}"
            if key.endswith(suffix):
                groups.add(key[: -len(suffix)])
                break
    return sorted(groups)


def candidate_keys_for_bucket(centers, group_keys, bucket):
    """The actual list of scorable "pop__direction__bucket" keys for a
    query item in this bucket -- one per group, but ONLY for groups that
    have a real (or already-fallback-substituted) cell at this bucket.
    Returns (candidate_keys, missing_groups) -- missing_groups had ZERO
    training items in that specific (group, bucket) cell (no key saved at
    all), and are reported separately rather than silently omitted, so the
    caller can log it once per item instead of the lookup just quietly
    coming up short."""
    candidate_keys = []
    missing_groups = []
    for group in group_keys:
        key = f"{group}__{bucket}"
        if key in centers:
            candidate_keys.append(key)
        else:
            missing_groups.append(group)
    return candidate_keys, missing_groups


def get_debut_candidate_keys(centers):
    """Which (group, bucket='debut') centers actually exist -- computed
    DYNAMICALLY from the loaded centers rather than a hardcoded 3-group
    list. The old DEBUT_CANDIDATE_KEYS constant (zero_popular__stable,
    medium_popular__increase, high_popular__increase) encoded a proof that
    only held for TRUE debut items (item_age == -1): a genuinely brand-new
    release can only reach those 3 groups, since its direction can only be
    'increase' or 'stable', never 'decrease' (see get_retrieved_direction).
    But s5_save_eight_groups.py's fix established that 'debut' bucket
    membership (n_points == 0) is NOT limited to true debut items -- an
    old, pre-collection-window item observed before its collection window
    opened can ALSO have zero real trend points with item_age far from -1,
    and such an item is classified through the NORMAL (non-debut) branch of
    get_retrieved_direction, so it can land in ANY of the 8 groups (see
    s5_save_eight_groups.py's module docstring, 'WHY DEBUT IS NO LONGER
    RESTRICTED TO A FIXED SET OF GROUPS'). Hardcoding the 3-group list here
    would silently exclude any group's debut cell outside that original
    proof -- this replaces it with whatever the actual saved profiles say."""
    return sorted(k for k in centers if k.endswith("__debut"))


# ----------------------------------------------------------------------------
# momentum calculation from trend_array
# ----------------------------------------------------------------------------
def calculate_normalized_momentum(trend_array, window=1):
    if not isinstance(trend_array, (list, np.ndarray)) or len(trend_array) < 2:
        return "unknown"

    arr = np.asarray(trend_array, dtype=float)
    w = min(window, len(arr) - 1)
    # average log1p difference over the trailing window, not just the last step
    log_vals = np.log1p(arr)
    diffs = np.diff(log_vals[-(w+1):])
    val = np.clip(diffs.mean(), -3.0, 3.0)
    return round(float(val), 3) if not np.isinf(val) and not pd.isna(val) else 0.0

# ---------------------------------------------------------------------------
# per-feature similarity functions
# ---------------------------------------------------------------------------
def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def pop_normalization(inf_value, center_value, gamma=1.0):
    """Exponential decay of the absolute difference -- used for every
    candidate feature except latest_momentum and embed_similarity."""
    return float(np.exp(-abs(float(center_value) - float(inf_value)) / gamma))


def embed_normalization(embed, center_vector):
    """Cosine similarity between item embedding and group centroid, scaled to [0, 1]."""
    distance = cosine(embed, center_vector)
    similarity = 1 - distance
    return (1 + similarity) / 2


def calculate_directional_distance(m1, m2, beta):
    """abs distance + a penalty when m1/m2 have opposite signs (LMNN-style hinge)."""
    abs_dist = np.abs(m1 - m2)
    directional_penalty = np.maximum(0.0, -(m1 * m2))
    return abs_dist + (beta * directional_penalty)


def compute_momentum_score(inf_mom, cand_mom, gamma_mom=1.0, beta_mom=1.0):
    if inf_mom == "unavailable" or cand_mom == "unavailable" or pd.isna(inf_mom) or pd.isna(cand_mom):
        return 0.5
    momentum_distance = calculate_directional_distance(float(inf_mom), float(cand_mom), beta_mom)
    return float(2 / (1 + np.exp(momentum_distance / gamma_mom)))


def _log1p(x):
    """Plain log1p for naturally nonnegative, right-skewed features --
    mirrors _apply_feature_transform's NONNEGATIVE_LOG_FEATURES handling
    in the correlation study, so production scoring and the correlation
    study operate in the same space."""
    return float(np.log1p(float(x)))


def raw_feature_score(feature, query_values, center, gamma_pop=1.0, gamma_mom=1.0, beta_mom=1.0):
    """Single feature's UNWEIGHTED similarity to one center -- the same
    per-feature routing composite_score uses internally, exposed standalone
    so callers can decompose a total score into its individual feature
    contributions (e.g. for margin/counterfactual analysis) instead of only
    getting the final weighted sum. Raises the same KeyErrors composite_score
    would for an undefined/unmapped feature. `center` must already be the
    bucket-specific cell (e.g. centers["zero_popular__decrease__maturing"]),
    not a whole-group entry -- see load_center_profiles."""
    if feature == "latest_momentum":
        return compute_momentum_score(
            query_values["latest_momentum"], center["latest_mom_median"], gamma_mom, beta_mom
        )
    if feature == "embed_similarity":
        return embed_normalization(query_values["embed_vector"], center["embed_vector"])

    center_key = FEATURE_TO_CENTER_KEY.get(feature)
    if center_key is None:
        raise KeyError(f"No center mapping defined for feature '{feature}'")
    if feature not in query_values or query_values[feature] is None:
        raise KeyError(
            f"Feature '{feature}' has no usable value in query_values for this item."
        )

    q_raw = query_values[feature]
    c_raw = center[center_key]

    if feature in SIGNED_LOG_FEATURES:
        q_val, c_val = _signed_log1p(q_raw), _signed_log1p(c_raw)
    elif feature in NONNEGATIVE_LOG_FEATURES:
        q_val, c_val = _log1p(q_raw), _log1p(c_raw)
    else:
        q_val, c_val = q_raw, c_raw

    return pop_normalization(q_val, c_val, gamma_pop)
# ---------------------------------------------------------------------------
# composite score -- rank-weighted by the LLM's selected_features order
# ---------------------------------------------------------------------------
def usable_features_equal_weight(feats, category):
    """INACTIVE-mode feature set: every candidate feature that's actually
    defined for this item (not None), in a fixed order, plus
    embed_similarity always (it's precomputed per-group and always
    available). Returns (feature_list, weight_list) with equal weights.

    category must be the LOWERCASE bucket name ('recent_single_point',
    'recent_multi_point', 'maturing') matching s5_save_eight_groups.py's
    get_history_depth_bucket -- see s14_retrieve_top1_centers.py, which no
    longer uses the old uppercase CATEGORY strings.

    RECENT_SINGLE_POINT is special-cased the same way it is in ACTIVE
    mode (just with equal instead of rank weights): a 1-point trend_array
    makes trend_min/trend_max trivial duplicates of latest_pop_value, and
    time_since_peak trivially 0 -- including them would silently dilute
    the two features that actually carry signal (latest_pop_value,
    embed_similarity) down to 1/6 weight each instead of 1/2. So this
    case always uses exactly those two, equally weighted, regardless of
    what compute_item_features happened to leave non-None.

    DEBUT is NOT handled here -- retrieve_top_1 forces
    embed_similarity-only for DEBUT before this function is ever called,
    in both modes."""
    if category == "recent_single_point":
        return ["latest_pop_value", "embed_similarity"], [0.5, 0.5]

    usable = [f for f in ALL_CANDIDATE_FEATURES if f == "embed_similarity" or feats.get(f) is not None]
    n = len(usable)
    weights = [1.0 / n] * n
    return usable, weights

def composite_score(selected_features, query_values, center,
                     gamma_pop=1.0, gamma_mom=1.0, beta_mom=1.0, weights=None):
    """
    selected_features: ranked list of feature names, e.g.
        ['latest_pop_value', 'trend_volatility', 'latest_momentum']
    query_values: dict of this query item's own feature values, using the
        same names as selected_features, plus 'embed_vector' if
        embed_similarity might be selected.
    center: one BUCKET-SPECIFIC entry from load_center_profiles()'s output
        dict, e.g. centers["zero_popular__decrease__maturing"] -- NOT a
        whole-group entry (there is no longer such a thing; see
        load_center_profiles).
    weights: optional explicit weight list, same length as
        selected_features, summing to 1.0 (or anything -- not enforced).
        Use this for the "feature selector INACTIVE" baseline mode, where
        every usable feature gets an equal weight (1/n each) instead of
        the rank-position weights below. If omitted (None), falls back to
        the fixed RANK_WEIGHTS lookup (the feature-selector-ACTIVE mode:
        3 features -> [0.50, 0.30, 0.20], 2 features -> [0.70, 0.30]).
    """
    n = len(selected_features)
    if weights is None:
        if n not in RANK_WEIGHTS:
            raise ValueError(
                f"composite_score expects 1-3 ranked features when weights is not "
                f"given explicitly, got {n}: {selected_features}"
            )
        weights = RANK_WEIGHTS[n]
    elif len(weights) != n:
        raise ValueError(
            f"weights length ({len(weights)}) must match selected_features length ({n})"
        )

    total = 0.0
    for feature, weight in zip(selected_features, weights):
        if feature == "latest_momentum":
            score = compute_momentum_score(
                query_values["latest_momentum"], center["latest_mom_median"], gamma_mom, beta_mom
            )
        elif feature == "embed_similarity":
            score = embed_normalization(query_values["embed_vector"], center["embed_vector"])
        else:
            center_key = FEATURE_TO_CENTER_KEY.get(feature)
            if center_key is None:
                raise KeyError(f"No center mapping defined for feature '{feature}'")
            if feature not in query_values or query_values[feature] is None:
                raise KeyError(
                    f"Feature '{feature}' was selected but has no usable value in query_values "
                    f"for this item -- check that the LLM didn't select an undefined/unknown feature."
                )
            elif feature in SIGNED_LOG_FEATURES:
                q_val = np.sign(query_values[feature]) * np.log1p(np.abs(query_values[feature]))
                c_val = np.sign(center[center_key]) * np.log1p(np.abs(center[center_key]))
                score = pop_normalization(q_val, c_val, gamma_pop)
            else:
                score = pop_normalization(np.log1p(query_values[feature]), np.log1p(center[center_key]), gamma_pop)
            # score = pop_normalization(query_values[feature], center[center_key], gamma_pop)
        total += weight * score
    return total

# ==================== ATTACH TO GROUND TRUTH (FOR POOL EVALUATION) ========
def popularity_threshold(pop_value):
    """Same rule used to assign pop_cluster elsewhere in the pipeline."""
    if pop_value == 0:
        return "zero_popular"
    elif pop_value > 0 and pop_value < 3:
        return "medium_popular"
    else:
        return "high_popular"

# def load_ground_truth(interpath, target_month):
#     inter_data = pd.read_csv(interpath)
#     inter_data['Time'] = pd.to_datetime(inter_data['Time'], unit="s")
#     inter_data['Month'] = inter_data['Time'].dt.strftime('%Y-%m')
#     grouped = inter_data.groupby(['MovieID', 'Month'])['UserID'].count().reset_index()
#     pivot = grouped.pivot(index="MovieID", columns='Month', values="UserID").reset_index()
#     pivot = pivot.rename(columns={'MovieID': 'movie_id'})
#     gt = pivot[['movie_id', target_month]].rename(columns={target_month: 'ground_truth_popularity_score'})
#     gt['ground_truth_popularity_score'] = gt['ground_truth_popularity_score'].fillna(0)
#     return gt


DOUBAN_TZ = ZoneInfo("UTC")

def load_ground_truth(interpath, target_month):
    inter_data = pd.read_csv(interpath)
    inter_data['Time'] = pd.to_datetime(inter_data['Time'], unit="s", utc=True).dt.tz_convert(DOUBAN_TZ)
    inter_data['Month'] = inter_data['Time'].dt.strftime('%Y-%m')
    grouped = inter_data.groupby(['MovieID', 'Month'])['UserID'].count().reset_index()
    pivot = grouped.pivot(index="MovieID", columns='Month', values="UserID").reset_index()
    pivot = pivot.rename(columns={'MovieID': 'movie_id'})
    gt = pivot[['movie_id', target_month]].rename(columns={target_month: 'ground_truth_popularity_score'})
    gt['ground_truth_popularity_score'] = gt['ground_truth_popularity_score'].fillna(0)
    return gt

def get_retrieved_direction(item_age,popularity_value, ground_truth_score) -> str:
    # NOTE: this checks item_age == 0 for the debut case, while
    # s5_save_eight_groups.py's get_retrieved_direction (the function that
    # actually produced the training labels) checks item_age == -1. Left
    # exactly as originally written -- not touched by this round's fix,
    # which was scoped to bucket/center keying, not ground-truth semantics.
    # Worth confirming which sentinel your query-side item_age actually
    # uses before trusting ground_truth_direction values from this function.
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

def attach_ground_truth_direction(query_df, interpath, target_month, id_col='movie_id'):
    """
    query_df needs: id_col, 'item_age', 'latest_pop_value'.
    Returns query_df with a 'ground_truth_direction' column attached.
    """
    gt = load_ground_truth(interpath, target_month)
    merged = query_df.merge(gt, left_on=id_col, right_on='movie_id', how='left')
    # fill NaN with 0 to mark items with absent interaction have 0 GT popularity value
    merged['ground_truth_popularity_score'] = merged['ground_truth_popularity_score'].fillna(0)
    merged['ground_truth_direction'] = merged.apply(
        lambda r: get_retrieved_direction(
            r['item_age'], r['latest_pop_value'], r['ground_truth_popularity_score']
        ),
        axis=1
    )
    merged['ground_truth_pop_cluster'] = merged['ground_truth_popularity_score'].apply(popularity_threshold)
    return merged

# ==================== EXPLANATION PARSING ====================
def preprocess_desensitized_predictive_signals(text):
    if pd.isna(text) or text == '':
        return text

    # Remove "PREDICTIVE SIGNALS:" header
    text = re.sub(r'EXPLANATORY FACTORS\s*:?\s*\n?', '', text, flags=re.IGNORECASE)

    # Desensitize raw "score of N" style mentions BEFORE any bullet/bracket
    # restructuring -- it operates on substrings this step doesn't touch,
    # so order relative to the steps below doesn't matter, but doing it
    # here keeps all content-level cleanup grouped before structural
    # formatting.
    text = desensitize_score_mentions(text)

    # Normalize unicode bullet • to "-"
    text = text.replace('•', '-')

    # Split on bullet markers (with OR without brackets)
    text = re.sub(r'\s*-\s*(\[?(?:Lifecycle|Metadata))', r'\n- \1', text, flags=re.IGNORECASE)

    # Normalize lines
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith('-'):
            line = f"- {line}"

        # Reformat: "- Lifecycle - Category - Impact explanation"
        # →         "- [Lifecycle: Category - Impact]: explanation"
        line = re.sub(
            r'-\s*(Lifecycle|Metadata)\s*-\s*(.+?)\s*-\s*(Positive|Negative|Neutral)\s*[:\]]?\s*',
            r'- [\1: \2 - \3]: ',
            line, flags=re.IGNORECASE
        )

        # Reformat: "- [Lifecycle - Category - Impact]" (dash instead of colon)
        # →         "- [Lifecycle: Category - Impact]:"
        line = re.sub(
            r'-\s*\[(Lifecycle|Metadata)\s*-\s*(.+?)\s*-\s*(Positive|Negative|Neutral)\]\s*:?\s*',
            r'- [\1: \2 - \3]: ',
            line, flags=re.IGNORECASE
        )

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def strip_valence_tags(text):
    """Removes the trailing ' - Positive]' / ' - Negative]' / ' - Neutral]'
    from bracketed factor tags, leaving the category/label but dropping the
    pre-judged valence -- e.g. '[Metadata: High-profile action cast - Positive]'
    becomes '[Metadata: High-profile action cast]'.
    """
    return re.sub(r'\s*-\s*(Positive|Negative|Neutral)\]', ']', text)
