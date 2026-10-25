"""
Douban Prediction File
"""
import json
import os
import re
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
_BULLET_FIX_PATTERN = re.compile(
    r"\[(Lifecycle|Metadata)\s*[-:]\s*(.+?)\s*[-:]\s*(Positive|Negative|Neutral)\]:+"
)
# Retrieval context produced by s17 with --g-tag G1 --e-variants E1 (the full G1E1 setup)
RETRIEVAL_CSV_TEMPLATE = (
    "processed_results/douban/retrieved_context_g1e1/"
    "douban_retrieval_context_per_query_embed_rerank_k{k}_a50_b50_gtprior0_G1E1.csv"
)
_TREND_ARRAY_SPLIT_PATTERN = re.compile(r"[,\s]+")

K_VALUES = [1,5,10]
MODEL = "gpt-4o-mini-2024-07-18"
OUTPUT_DIR = "prediction_output/gpt4omini_douban"
NO_RETRIEVAL_FALLBACK = "No similar past items provided."
# INFERENCE_MONTH_STR = "2019-11"  # test-set prediction time, fixed for this whole run
INFERENCE_MONTH_STR = '2019-11'
# SYSTEM_PROMPT_PATH removed -- system_prompt is hardcoded in prepare_batch_input, not loaded from disk
# SMOKE_TEST_MOVIE_IDS = [
#     27119724, 30166972, 30402296, 27109633, 25887288,
#     30299311, 26786669, 30318116, 30362186, 33400376,
#     30390144, 3097572, 26794435, 30413052, 26612258,
#     27010768, 6981153, 26715636, 30272143, 25779217,
# ]
# ── END CONFIG ────────────────────────────────────────────────────────────

def parse_center_name(center_name):
    """'high_popular__decrease' -> ('high_popular', 'decrease'). Returns
    (None, None) for anything unparseable (e.g. a scoring error row)."""
    if not isinstance(center_name, str) or "__" not in center_name:
        return None, None
    pop_cluster, direction = center_name.split("__", 1)
    return pop_cluster, direction

def sample_equal_per_direction(
    df: pd.DataFrame,
    target_n: int,
    target_high_pop: int = 10,
    cluster_col: str = 'ground_truth_pop_cluster',
    high_pop_val: str = 'high_popular',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Samples target_n rows equalized across ground_truth_direction,
    enforcing exactly `target_high_pop` total rows with high popularity cluster.
    """
    directions = ['increase', 'decrease', 'stable']
    n_dirs = len(directions)

    high_per_dir = target_high_pop // n_dirs
    high_remainder = target_high_pop % n_dirs

    tot_per_dir = target_n // n_dirs
    tot_remainder = target_n % n_dirs

    sampled_chunks = []

    for i, d in enumerate(directions):
        n_high_needed = high_per_dir + (1 if i < high_remainder else 0)
        n_total_needed = tot_per_dir + (1 if i < tot_remainder else 0)
        n_normal_needed = n_total_needed - n_high_needed

        sub_dir = df[df['ground_truth_direction'] == d]

        sub_high = sub_dir[sub_dir[cluster_col] == high_pop_val]
        sub_normal = sub_dir[sub_dir[cluster_col] != high_pop_val]

        if len(sub_high) <= n_high_needed:
            sampled_high = sub_high
        else:
            sampled_high = sub_high.sample(n=n_high_needed, random_state=random_state)

        if len(sub_normal) <= n_normal_needed:
            sampled_normal = sub_normal
        else:
            sampled_normal = sub_normal.sample(n=n_normal_needed, random_state=random_state)

        sampled_chunks.extend([sampled_high, sampled_normal])

    return pd.concat(sampled_chunks).reset_index(drop=True)


def normalize_explanatory_factor_bullets(text):
    """Normalize any '[Domain <sep> Category <sep> Impact]:<:>' variant to the
    canonical '[Domain - Category - Impact]:'. Idempotent: the canonical form
    matches too (dash is one of the accepted separators) but replaces with itself."""
    if not isinstance(text, str):
        return text
    return _BULLET_FIX_PATTERN.sub(r"[\1 - \2 - \3]:", text)


def load_retrieval_csv(k):
    path = RETRIEVAL_CSV_TEMPLATE.format(k=k)
    df = pd.read_csv(path)

    df["trend_array"] = df["trend_array"].apply(
        lambda x: np.array([int(i) for i in _TREND_ARRAY_SPLIT_PATTERN.split(x.strip("[]").strip()) if i])
        if isinstance(x, str) else x
    )

    return df

# ── Friend-format alignment helpers ─────────────────────────────────────

def _fallback_na(value):
    """Return 'NA' for missing/empty values, else the value as-is (matching
    his observed 'written by NA' fallback convention)."""
    if value is None:
        return "NA"
    if isinstance(value, float) and pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return "NA"
    return text


def capitalize_names(text):
    """Title-case a comma-separated list of person names, e.g.
    'frank miller,robert rodriguez' -> 'Frank Miller, Robert Rodriguez'.
    Leaves 'NA' untouched. Handles apostrophes/hyphens correctly since
    Python's str.title() treats them as word boundaries (o'neal -> O'Neal,
    gordon-levitt -> Gordon-Levitt)."""
    if not isinstance(text, str) or text.strip().upper() == "NA":
        return text
    names = [name.strip().title() for name in text.split(",")]
    return ", ".join(name for name in names if name)


def build_movie_description(row):
    """Reproduce the friend's exact MovieDescription template from our
    structured columns:
      movie_name, release_date, genre, description (his "synopsis"),
      language, director, writers, top5_actors, producers.
    We have no music/composer column, so that clause always renders "NA"
    (following the same NA-fallback convention he uses for a missing writer).

    Example target:
    "The movie Maleficent: Mistress of Evil (2019) is a Fantasy, Adventure
    film released in 2019-10-17. In this movie, <synopsis>. It is in
    English, directed by Joachim Rønning, and written by NA. And the main
    actors (top-5) include Angelina Jolie, ... . The movie is produced by
    ... and the music is produced by ...."
    """
    movie_name = _fallback_na(row.get("movie_name"))
    release_date = _fallback_na(row.get("release_date"))
    year = release_date[:4] if release_date != "NA" and len(release_date) >= 4 else "NA"
    genre = _fallback_na(row.get("genre"))
    synopsis = _fallback_na(row.get("description"))
    language = _fallback_na(row.get("language"))
    director = capitalize_names(_fallback_na(row.get("director")))
    writers = capitalize_names(_fallback_na(row.get("writers")))
    top5_actors = _fallback_na(row.get("top5_actors"))
    producers = _fallback_na(row.get("producers"))
    music = _fallback_na(row.get("music"))  # no music/composer column available in this pipeline

    return (
        f"The movie {movie_name} is a {genre} film released in {release_date}. "
        f"In this movie, {synopsis}. "
        f"It is in {language}, directed by {director}, and written by {writers}. "
        f"And the main actors (top-5) include {top5_actors}. "
        f"The movie is produced by {producers} and the music is produced by {music}."
    )

def _month_label(date_value):
    """'YYYY-MM' label for a date-like value, or None if missing/unparsable."""
    if pd.isna(date_value):
        return None
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M"))
    except (ValueError, TypeError):
        return None


def _month_before(date_value):
    """'YYYY-MM' label for the calendar month immediately preceding date_value."""
    if pd.isna(date_value):
        return None
    try:
        return str(pd.Period(pd.Timestamp(date_value), freq="M") - 1)
    except (ValueError, TypeError):
        return None

def build_target_trend_line(trend_array, release_date, inference_month_str):
    """Same rendering as the retrieved-item cards' Trend line -- a date
    window (release month through the month before InferenceTimestamp)
    plus a plain comma-separated value sequence -- built from trend_array
    instead of the colon-string. Matching the retrieved items' format here
    matters: the old {timestamp, popularity_score} dict rendering was
    implicated in own_estimate just echoing the last listed value rather
    than reasoning over the trend, and giving the target a differently-shaped
    history than its comps risked the model treating them as two different
    kinds of object instead of the same kind of sequence.

    Returns None for a debut item (empty or [-1]-sentinel trend_array) --
    callers should render an explicit "No historical trend" note in that
    case rather than an empty list literal.
    """
    trend_list = trend_array.tolist() if hasattr(trend_array, "tolist") else list(trend_array)
    if len(trend_list) == 0 or (len(trend_list) == 1 and trend_list[0] == -1):
        return None

    start_label = _month_label(release_date)
    end_label = _month_before(inference_month_str)

    if start_label and end_label:
        window = f", {start_label}" if start_label == end_label else f", {start_label} to {end_label}"
        window += " (monthly)"
    else:
        window = ""
    values_str = ", ".join(map(str, trend_list))
    label = f"Trend{window}" if window else "Trend (full history)"

    return f"{label}: {values_str}"

def build_target_item_block(row, inference_month_str):
    movie_description_text = build_movie_description(row)

    if row['item_age'] == -1:
        # Debut item -- no release history yet, same convention as the
        # retrieved-item debut cards (which omit the Trend line entirely).
        popularity_history_text = "Debut item — no historical trend available"
    else:
        trend_line = build_target_trend_line(
            row['trend_array'], row.get('release_date'), inference_month_str
        )
        popularity_history_text = trend_line if trend_line is not None else "No historical trend"

    return f"""
<MovieDescription>{movie_description_text}</MovieDescription>

<PopularityHistory>{popularity_history_text}</PopularityHistory>

<InferenceTimestamp>{inference_month_str}</InferenceTimestamp>
"""

def prepare_batch_input(df, model, variant_tag, inference_month_str,
                         include_retrieval=True, output_dir="."):
# Simple Prompt V5 (stripped direction and popularity tiers) + no indication which one is most similar, but sorted ascending
    system_prompt = """You are an expert movie popularity analyst. Your task is to predict a movie's popularity score at a given point in time, based on its own
descriptive features, its historical popularity trajectory (when available), and a set of retrieved past movies with similar description and popularity patterns.

### About <PopularityHistory>:
<PopularityHistory> covers the target's own monthly popularity values from
release through one month before <InferenceTimestamp> -- it does not
include the value at <InferenceTimestamp> itself, which is what you are
being asked to predict.

### About <SimilarPastItems>:
<SimilarPastItems> shows past movies retrieved for similar description and
popularity pattern -- they will not necessarily match the target's outcome. 
For non-debut items, each item's own trend line covers its own history up through one month before its own outcome month; its outcome is stated only as a relative percentage change from that trend's last value (never the raw outcome number itself), showing how much the observed popularity moved. 
For debut items, neither a trend line nor an outcome value is provided, since there is no prior popularity history — only its own movie description and explanatory factors are available for these cases.

Steps:
1. First, estimate the target's popularity score using only <MovieDescription>
   and <PopularityHistory> (when available). It's a specific number, not a category label.
2. Then review <SimilarPastItems>. For each retrieved item, check whether
   its trend trajectory and explanatory factors -- covering both its own
   movie description and its own popularity history -- are actually
   applicable to the target item. Use this review to adjust your step 1
   estimate up, down, or keep it stable. Repeated factors across items
   count as one signal, not reinforcement.

Output a single, valid JSON object with exactly these two keys:

- `reasoning`: must cover two things, in order -- (1) your step 1 initial
  estimate and what it's based on, and (2) your justification for the
  final predict_popularity_score, including whether and why
  <SimilarPastItems> moved it up, down, or left it unchanged from that
  initial estimate.
- `predict_popularity_score`: your final predicted popularity score at the
  given <InferenceTimestamp>. Non-negative integer, formatted as a string.

Output only this JSON object, with nothing before or after.

### Input format:

<MovieDescription>...</MovieDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
<SimilarPastItems>...</SimilarPastItems>
"""

    print(f"Preparing batch: variant={variant_tag} (include_retrieval={include_retrieval}), {len(df)} rows")

    batch_requests = []
    custom_ids = []

    for idx, row in df.iterrows():
        custom_id = f"pred_{variant_tag}_request-{idx + 1:05d}"
        custom_ids.append(custom_id)

        if include_retrieval:
            retrieved_section = f"<SimilarPastItems>\n{row['retrieved_context']}\n</SimilarPastItems>"
        else:
            retrieved_section = f"<SimilarPastItems>\n{NO_RETRIEVAL_FALLBACK}\n</SimilarPastItems>"

        target_item_block = build_target_item_block(row, inference_month_str)

        user_content = f"""{target_item_block}
        
        {retrieved_section}
        """

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": 4096,
            "temperature": 0.1,
            "top_p": 1.0,
            "stream": False,
        }
# earliest setup was t = 0.4 top_p default

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        batch_requests.append(request)

    mapping_df = df.copy()
    mapping_df["custom_id"] = custom_ids
    if not include_retrieval and "retrieved_context" in mapping_df.columns:
        mapping_df = mapping_df.drop(columns=["retrieved_context"])

    os.makedirs(output_dir, exist_ok=True)
    mapping_path = f"{output_dir}/mapping_{variant_tag}.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved mapping ({len(mapping_df)} rows) -> {mapping_path}")
    print(f"Created JSONL batch with {len(batch_requests)} requests")
    return batch_requests


def save_jsonl(requests, variant_tag, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/pred_batch_{variant_tag}.jsonl"
    with open(output_file, "w") as f:
        for item in requests:
            f.write(json.dumps(item) + "\n")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {len(requests)} requests -> {output_file} ({size_mb:.1f} MB)")
    return output_file


def main():
    for k in K_VALUES:
        df = load_retrieval_csv(k)
        # df = df[df['movie_id'].isin(SMOKE_TEST_MOVIE_IDS)].copy()
        variant_tag = f"k{k}_a50_b50_sig1_gtcase1_gtprior1"
        requests = prepare_batch_input(
            df, MODEL, variant_tag, INFERENCE_MONTH_STR,
            include_retrieval=True, output_dir=OUTPUT_DIR,
        )
        save_jsonl(requests, variant_tag, OUTPUT_DIR)

    df_no_retrieval = load_retrieval_csv(K_VALUES[0])
    # df_no_retrieval = df_no_retrieval[df_no_retrieval['movie_id'].isin(SMOKE_TEST_MOVIE_IDS)].copy()
    requests_no_retrieval = prepare_batch_input(
        df_no_retrieval, MODEL, "no_retrieval", INFERENCE_MONTH_STR,
        include_retrieval=False, output_dir=OUTPUT_DIR,
    )
    save_jsonl(requests_no_retrieval, "no_retrieval", OUTPUT_DIR)


if __name__ == "__main__":
    main()
