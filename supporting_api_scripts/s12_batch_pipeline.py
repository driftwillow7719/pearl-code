"""
OpenAI Batch API pipeline -- submit, check, cancel, and download, all in
one file.
Usage:
    python s12_batch_pipeline.py submit  --jsonl path/to/requests.jsonl --tag my_batch
    python s12_batch_pipeline.py status  --tag my_batch
    python s12_batch_pipeline.py cancel  --tag my_batch
    python s12_batch_pipeline.py download --tag my_batch --output-dir out/ [--wait] \\
        [--mapping-csv processed_results/260821/260821_fs_val/feature_selection_mapping.csv]

Requires: pip install openai python-dotenv pandas
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


BATCH_META_SUFFIX = "_batch_meta.json"  # {tag}_batch_meta.json in the current directory


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _client():
    load_dotenv()
    return OpenAI()


def _meta_path(tag):
    return f"{tag}{BATCH_META_SUFFIX}"


def _load_meta(tag):
    path = _meta_path(tag)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No batch metadata found at {path} -- submit a batch with this "
            f"tag first (`submit --tag {tag}`), or check the tag spelling."
        )
    with open(path) as f:
        return json.load(f)


def _save_meta(tag, meta):
    with open(_meta_path(tag), "w") as f:
        json.dump(meta, f, indent=2)


def clean_json_text(text):
    """Strip ```json ... ``` / ``` ... ``` fences some models wrap output in."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

def insert_missing_comma_between_strings(text):
    """Fix a missing ',' between two adjacent JSON string tokens, e.g.
    '..."final estimate of 2."\n  "predict_popularity_score": "2"' -- the
    model closed one key's string value and opened the next key's string
    without the required comma in between. Only fires when the gap between
    the two quotes is PURE whitespace (so a legitimate empty string "" is
    never touched -- that has a zero-width gap -- and an already-valid
    '"...", \n "next_key"' is never touched either, since the comma there
    breaks the whitespace-only run)."""
    return re.sub(r'"(\s+)"', r'",\1"', text)

def csv_unescape_doubled_quotes(text):
    """Undo RFC 4180 CSV quote-doubling ('""' -> '"') if the text still has it.

    Only fires when doubled quotes are actually present, so it's a no-op on
    text that's already been through a real CSV reader (csv.reader /
    pandas.read_csv already do this unescaping for you -- if you're seeing
    doubled quotes at this point, it usually means the text arrived via
    manual line-splitting instead of proper CSV parsing, and this layer
    should really be fixed upstream)."""
    if '""' not in text:
        return text
    return text.replace('""', '"')


def strip_trailing_commas(text):
    """Fix the one malformed-JSON pattern we've actually seen models emit:
    a trailing comma right before a closing '}' or ']' (valid in Python/JS,
    not valid JSON), e.g. '..."reasoning": "...",\\n}'. Only removes a comma
    that is immediately followed (ignoring whitespace) by a closing bracket
    -- never touches a comma that's actually separating two real items."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def escape_unescaped_inner_quotes(text):
    """Repair the common weak-model failure of emitting literal '"' characters
    inside a JSON string value (e.g. quoting a movie title in `reasoning`)
    instead of backslash-escaping them, which breaks json.loads with errors
    like "Expecting ',' delimiter" partway through the string.

    Walks the text tracking whether we're inside a string. Once inside a
    string, a '"' is only treated as the REAL closing quote if -- skipping
    any whitespace -- it's immediately followed by something that can
    legally follow a JSON string (',', ':', '}', ']', or end of input).
    Otherwise it's an embedded quote and gets escaped in place. Existing
    escape sequences ('\\"', '\\n', etc.) are passed through untouched."""
    out = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and in_string and i + 1 < n:
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ",:}]":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def repair_unterminated_trailing_string(text):
    """Fix a rarer corruption pattern: the model's output ends with a run of
    stray tokens (seen in practice: full-width CJK punctuation like
    '】【。') instead of properly closing the last string value,
    e.g. '..."reasoning": "...older low-level items.】【。}' --
    no closing '"' before the final '}'. Only fires if the text ends with a
    run of '}'/']' brackets with no closing quote immediately before them;
    strips the trailing garbage back to the last normal-prose character and
    inserts the missing '"'. Deliberately conservative: does nothing if the
    text already looks properly terminated, to avoid corrupting otherwise-
    valid JSON."""
    text = text.rstrip()
    if not text.endswith("}"):
        return text

    i = len(text)
    while i > 0 and text[i - 1] in "}]":
        i -= 1
    tail_close = text[i:]
    body = text[:i]

    if body.endswith('"'):
        return text

    stripped = body.rstrip()
    while stripped and (ord(stripped[-1]) > 0x2000 or stripped[-1] in "()[]{}<>"):
        stripped = stripped[:-1]

    if stripped == body:
        return text

    return stripped.rstrip() + '"' + tail_close


def parse_json_safe(text):
    """Progressively repair and parse. Repairs are applied cumulatively (each
    stage builds on the previous one's output) since real-world corruption
    is often compound -- e.g. CSV-doubled quotes AND unescaped inner quotes
    both need fixing together before some rows will parse. Order matters:
    CSV unescaping must happen before inner-quote escaping, since until the
    doubled quotes are collapsed to single quotes there's no way to tell a
    structural delimiter from embedded content."""
    stage = clean_json_text(text)

    # try raw first -- don't touch text that's already valid
    try:
        return json.loads(stage), None
    except (json.JSONDecodeError, TypeError) as e:
        first_error = str(e)

    stages = [
        csv_unescape_doubled_quotes,
        insert_missing_comma_between_strings,   # new — must come before the next one
        escape_unescaped_inner_quotes,
        strip_trailing_commas,
        repair_unterminated_trailing_string,
    ]

    for fn in stages:
        stage = fn(stage)
        try:
            return json.loads(stage), None
        except (json.JSONDecodeError, TypeError):
            continue

    return {}, first_error

def apply_batch_results(mapping_df, batch_results_by_custom_id):
    """After running the JSONL batch, call this to fold results back onto
    the mapping CSV. batch_results_by_custom_id: {custom_id: parsed JSON
    response dict, e.g. {"selected_features": [...], "rationale": "..."}}.
    recent_single_point rows now DO go through the LLM (2-feature
    selection, ranked by importance) -- they're looked up by custom_id
    exactly like every other non-debut bucket, no fixed-result shortcut
    anymore. debut rows (custom_id is None, history_depth_bucket ==
    'debut') are left untouched here -- they never reach the LLM at all;
    they're resolved separately via resolve_debut_group's nearest-pooled-
    embed_similarity lookup, not by selected_features/composite_score."""
    def resolve(row):
        if row["history_depth_bucket"] == "debut":
            return None
        return batch_results_by_custom_id.get(row["custom_id"])

    mapping_df = mapping_df.copy()
    results = mapping_df.apply(resolve, axis=1)
    mapping_df["selected_features"] = results.apply(
        lambda r: r["selected_features"] if r else None
    )
    mapping_df["selection_rationale"] = results.apply(
        lambda r: r["rationale"] if r else None
    )
    return mapping_df
# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def submit_batch(jsonl_path, tag, description="", completion_window="24h"):
    client = _client()

    print(f"Uploading {jsonl_path} ...")
    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    print(f"  file_id: {uploaded.id}")

    print("Creating batch job ...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": description or tag, "source": "batch_pipeline"},
    )

    meta = {
        "tag": tag,
        "batch_id": batch.id,
        "file_id": uploaded.id,
        "jsonl_source": os.path.abspath(jsonl_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status_at_submit": batch.status,
    }
    _save_meta(tag, meta)

    print(f"Batch created: {batch.id} (status: {batch.status})")
    print(f"Metadata saved -> {_meta_path(tag)}")
    return meta


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def get_status(tag, client=None):
    client = client or _client()
    meta = _load_meta(tag)
    batch = client.batches.retrieve(meta["batch_id"])

    comp = batch.request_counts.completed
    total = batch.request_counts.total
    fail = batch.request_counts.failed
    progress = (comp / total * 100) if total else 0
    icon = "done" if batch.status == "completed" else ("failed" if batch.status in ("failed", "expired", "cancelled") else "in progress")

    print(f"[{tag}] batch_id={meta['batch_id']}")
    print(f"  status: {batch.status} ({icon})")
    print(f"  progress: {comp}/{total} ({progress:.1f}%), failed: {fail}")
    return batch


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

def cancel_batch(tag, client=None):
    """Cancel an in-progress batch. OpenAI only bills for requests that
    actually completed before cancellation -- anything still queued or
    in-flight is dropped, unbilled. Cancellation isn't instant: the batch
    moves to 'cancelling' first, then 'cancelled' once OpenAI finishes
    tearing it down (usually within a few minutes) -- this function
    reports whatever status comes back immediately rather than blocking
    until it settles, matching how get_status already reports whatever
    state is current at the moment you ask.

    Calling this on a batch that's already completed/failed/expired/
    cancelled is a no-op on OpenAI's side; it just returns that terminal
    status back to you instead of erroring, which is what happened here."""
    client = client or _client()
    meta = _load_meta(tag)

    batch = client.batches.retrieve(meta["batch_id"])
    if batch.status in ("completed", "failed", "expired", "cancelled"):
        print(f"[{tag}] batch_id={meta['batch_id']} is already in a terminal state "
              f"({batch.status}) -- nothing to cancel.")
        return batch

    comp = batch.request_counts.completed
    total = batch.request_counts.total
    print(f"[{tag}] batch_id={meta['batch_id']} -- cancelling "
          f"(was: {batch.status}, {comp}/{total} completed so far)...")

    batch = client.batches.cancel(meta["batch_id"])

    meta["cancelled_at"] = datetime.now(timezone.utc).isoformat()
    meta["status_at_cancel"] = batch.status
    _save_meta(tag, meta)

    print(f"  status: {batch.status}")
    print(f"  {comp}/{total} request(s) had already completed and will still be billed; "
          f"the rest were dropped, unbilled.")
    print(f"  Note: cancellation can take a few minutes to fully settle -- "
          f"re-run `status --tag {tag}` shortly to confirm it reaches 'cancelled'.")
    return batch

# ---------------------------------------------------------------------------
# retry -- rebuild a JSONL of just the failed requests and resubmit them as
# a new batch under a new tag
# ---------------------------------------------------------------------------

def collect_failed_custom_ids(tag, output_dir="."):
    """Gather every custom_id that needs retrying for `tag`, from BOTH
    failure sources that show up in this pipeline:

    1. Request-level failures in {tag}_errors.csv -- the API itself
       rejected or errored on the request (rate limit, content filter,
       timeout, etc.). These never produced a response at all.
    2. Response-level failures in {tag}.csv -- the API call succeeded, but
       the model's own output was malformed JSON that parse_json_safe
       couldn't repair (marked with a '_parse_error' column). These DID
       get billed, but there's no usable result to keep.

    Returns (custom_ids, breakdown) where breakdown is a dict counting how
    many came from each source, so you can report both numbers rather than
    a single opaque total."""
    custom_ids = set()
    breakdown = {"request_errors": 0, "parse_errors": 0}

    err_csv_path = os.path.join(output_dir, f"{tag}_errors.csv")
    if os.path.exists(err_csv_path):
        err_df = pd.read_csv(err_csv_path)
        ids = err_df["custom_id"].dropna().tolist()
        custom_ids.update(ids)
        breakdown["request_errors"] = len(ids)

    results_csv_path = os.path.join(output_dir, f"{tag}.csv")
    if os.path.exists(results_csv_path):
        results_df = pd.read_csv(results_csv_path)
        if "_parse_error" in results_df.columns:
            parse_failed = results_df[results_df["_parse_error"].notna()]
            ids = parse_failed["custom_id"].dropna().tolist()
            custom_ids.update(ids)
            breakdown["parse_errors"] = len(ids)

    return custom_ids, breakdown


def build_retry_jsonl(tag, output_dir=".", retry_tag=None):
    """Look up the original submitted JSONL (path saved in
    {tag}_batch_meta.json at submit time) and pull out just the request
    lines whose custom_id needs retrying, writing them to a fresh JSONL
    ready to resubmit. Returns (retry_jsonl_path, custom_ids, breakdown),
    or (None, set(), breakdown) if there's nothing to retry."""
    meta = _load_meta(tag)
    custom_ids, breakdown = collect_failed_custom_ids(tag, output_dir)

    if not custom_ids:
        print(f"[{tag}] no failed requests found in {tag}_errors.csv or "
              f"{tag}.csv -- nothing to retry.")
        return None, custom_ids, breakdown

    source_path = meta["jsonl_source"]
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Original request JSONL not found at {source_path} (recorded "
            f"in {_meta_path(tag)} at submit time) -- can't rebuild retry "
            f"requests without it. Pass the original file path directly if "
            f"it's moved since."
        )

    retry_tag = retry_tag or f"{tag}_retry"
    retry_jsonl_path = os.path.join(output_dir, f"{retry_tag}.jsonl")

    matched = 0
    with open(source_path, "r") as src, open(retry_jsonl_path, "w") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("custom_id") in custom_ids:
                out.write(line + "\n")
                matched += 1

    if matched < len(custom_ids):
        missing = len(custom_ids) - matched
        print(f"⚠️  {missing} failed custom_id(s) from {tag} were not found "
              f"in {source_path} -- they won't be retried. This usually "
              f"means the source JSONL has changed since the original "
              f"submission, or custom_ids don't match exactly.")

    print(f"[{tag}] {matched} failed request(s) "
          f"({breakdown['request_errors']} request errors, "
          f"{breakdown['parse_errors']} parse errors) -> {retry_jsonl_path}")
    return retry_jsonl_path, custom_ids, breakdown


def retry_failed(tag, output_dir=".", retry_tag=None, description="", client=None):
    """Build a JSONL of tag's failed requests and submit it as a new batch.
    The new batch's metadata records `retry_of` so you can trace the chain
    back, and how many attempts deep this is (`retry_attempt`), in case a
    retry itself has failures that need a further retry."""
    client = client or _client()
    retry_tag = retry_tag or f"{tag}_retry"

    retry_jsonl_path, custom_ids, breakdown = build_retry_jsonl(
        tag, output_dir=output_dir, retry_tag=retry_tag
    )
    if retry_jsonl_path is None:
        return None

    meta = submit_batch(
        retry_jsonl_path, retry_tag,
        description=description or f"retry of {tag}",
    )

    # tag the new batch's metadata with retry lineage, walking back through
    # any prior retries to compute how many attempts deep this chain is
    parent_meta = _load_meta(tag)
    meta["retry_of"] = tag
    meta["retry_attempt"] = parent_meta.get("retry_attempt", 0) + 1
    meta["retried_custom_id_count"] = len(custom_ids)
    _save_meta(retry_tag, meta)

    print(f"Retry batch submitted as '{retry_tag}' (attempt "
          f"#{meta['retry_attempt']} in this retry chain) -- "
          f"check with `status --tag {retry_tag}`.")
    return meta

# ---------------------------------------------------------------------------
# finalize -- fold the parsed per-item results (still in memory, as a list
# of dicts, NOT re-read from the results CSV) onto the mapping CSV via
# apply_batch_results, filling debut/recent_single_point fixed values too
# ---------------------------------------------------------------------------

def finalize_mapping_from_rows(rows, mapping_csv_path, output_dir, tag):
    """rows: the list of {"custom_id": ..., "selected_features": [...],
    "rationale": "...", maybe "_parse_error"/"_raw_content"} dicts that
    download_and_parse already built from the raw batch output -- passed
    in directly (not re-read from {tag}.csv) specifically to avoid a CSV
    round-trip, since pandas writes a Python list column like
    selected_features as its str() repr, not valid JSON/Python-literal-safe
    text, so re-parsing it back out of a CSV is fragile. Working from the
    in-memory rows sidesteps that entirely.

    Returns the enriched mapping DataFrame and also writes it to
    {output_dir}/{tag}_final_mapping.csv."""
    batch_results_by_custom_id = {
        row["custom_id"]: {
            "selected_features": row.get("selected_features"),
            "rationale": row.get("rationale"),
        }
        for row in rows
        if "_parse_error" not in row  # a malformed response has no usable
                                       # selected_features/rationale to fold in;
                                       # leave those rows to resolve() below,
                                       # which will report them via
                                       # batch_results_by_custom_id.get(...) -> None
    }

    mapping_df = pd.read_csv(mapping_csv_path)
    enriched_df = apply_batch_results(mapping_df, batch_results_by_custom_id)

    final_path = os.path.join(output_dir, f"{tag}_final_mapping.csv")
    enriched_df.to_csv(final_path, index=False)

    n_missing = enriched_df["selected_features"].isna().sum()
    if n_missing:
        print(f"⚠️  {n_missing} row(s) in the final mapping still have no "
              f"selected_features -- likely custom_ids with a parse error "
              f"or a request that failed/errored in the batch (check "
              f"{tag}_errors.csv and the '_parse_error' rows, if any).")

    print(f"Finalized mapping ({len(enriched_df)} rows) -> {final_path}")
    return enriched_df


# ---------------------------------------------------------------------------
# download + parse (one pass)
# ---------------------------------------------------------------------------

def download_and_parse(tag, output_dir=".", poll_interval_s=60, wait=False, client=None,
                        mapping_csv_path=None):
    """mapping_csv_path: optional. If given, after parsing the batch's raw
    responses, also folds them onto that mapping CSV (via
    finalize_mapping_from_rows/apply_batch_results) and writes
    {output_dir}/{tag}_final_mapping.csv -- the one CSV with
    selected_features/selection_rationale filled for every row, including
    debut/recent_single_point's fixed values. If omitted, this function
    behaves exactly as before: just the flat {tag}.csv of parsed
    per-request responses."""
    client = client or _client()
    meta = _load_meta(tag)
    os.makedirs(output_dir, exist_ok=True)

    terminal_statuses = ("completed", "failed", "expired", "cancelled")

    batch = client.batches.retrieve(meta["batch_id"])
    if wait:
        while batch.status not in terminal_statuses:
            comp = batch.request_counts.completed
            total = batch.request_counts.total
            print(f"  [{tag}] {batch.status} ({comp}/{total}) -- polling again in {poll_interval_s}s")
            time.sleep(poll_interval_s)
            batch = client.batches.retrieve(meta["batch_id"])

    if batch.status not in terminal_statuses:
        print(f"Batch not ready (status: {batch.status}). Re-run with --wait, or check again later.")
        return None

    if not batch.output_file_id and not batch.error_file_id:
        print(f"Batch ended with status '{batch.status}' and produced no output or error file -- nothing to download.")
        return None

    if batch.status != "completed":
        comp = batch.request_counts.completed
        total = batch.request_counts.total
        print(f"Note: batch status is '{batch.status}' ({comp}/{total} completed before it ended) -- "
              f"parsing whatever partial results exist.")

    raw_jsonl_path = os.path.join(output_dir, f"{tag}_raw.jsonl")
    csv_path = os.path.join(output_dir, f"{tag}.csv")
    err_csv_path = os.path.join(output_dir, f"{tag}_errors.csv")

    rows = []
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).content
        with open(raw_jsonl_path, "wb") as f:
            f.write(content)

        n_parse_failed = 0
        with open(raw_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                body = data.get("response", {}).get("body", {})
                choices = body.get("choices") or []
                if not choices:
                    continue
                message_content = choices[0]["message"]["content"]
                parsed, parse_error = parse_json_safe(message_content)
                row = {"custom_id": data["custom_id"], **parsed}
                if parse_error is not None:
                    n_parse_failed += 1
                    row["_parse_error"] = parse_error
                    row["_raw_content"] = message_content
                rows.append(row)

        if n_parse_failed:
            print(f"⚠️  {n_parse_failed} response(s) had malformed JSON that couldn't be "
                  f"parsed even after trailing-comma repair -- these rows have empty "
                  f"selected_features/reasoning but a '_parse_error' and '_raw_content' "
                  f"column so you can inspect what the model actually returned.")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"Parsed {len(df)} responses -> {csv_path}")
        else:
            print("No successful responses to parse.")
    else:
        print("No output file (0 successful requests).")

    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).content
        err_jsonl_path = os.path.join(output_dir, f"{tag}_errors_raw.jsonl")
        with open(err_jsonl_path, "wb") as f:
            f.write(err_content)

        errors = []
        for line in err_content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)

            top_error = data.get("error") or {}
            response = data.get("response") or {}
            body = response.get("body") or {}
            body_error = body.get("error") or {}

            error_code = top_error.get("code") or body_error.get("code")
            error_message = top_error.get("message") or body_error.get("message")

            errors.append({
                "custom_id": data.get("custom_id"),
                "status_code": response.get("status_code"),
                "error_code": error_code,
                "error_type": body_error.get("type") or top_error.get("type"),
                "error_message": error_message,
                "_raw_line": None if (error_code or error_message) else json.dumps(data),
            })

        if errors:
            err_df = pd.DataFrame(errors)
            err_df.to_csv(err_csv_path, index=False)
            print(f"{len(errors)} failed requests -> {err_csv_path}")
            print(err_df["error_code"].value_counts().to_string())
    else:
        print("No errors reported.")

    if mapping_csv_path and rows:
        finalize_mapping_from_rows(rows, mapping_csv_path, output_dir, tag)

    return csv_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API pipeline: submit / status / cancel / download")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="upload a JSONL file and create a batch job")
    p_submit.add_argument("--jsonl", required=True, help="path to the batch input JSONL")
    p_submit.add_argument("--tag", required=True, help="short identifier for this batch (used to name the metadata/output files)")
    p_submit.add_argument("--description", default="", help="optional description stored in batch metadata")

    p_status = sub.add_parser("status", help="check a batch's current status")
    p_status.add_argument("--tag", required=True)

    p_cancel = sub.add_parser("cancel", help="cancel an in-progress batch (no charge for anything not yet completed)")
    p_cancel.add_argument("--tag", required=True)

    p_download = sub.add_parser("download", help="download and parse a completed batch's results")
    p_download.add_argument("--tag", required=True)
    p_download.add_argument("--output-dir", default=".", help="directory to save the parsed CSV / raw JSONL / errors into")
    p_download.add_argument("--wait", action="store_true", help="poll until the batch completes, then download automatically")
    p_download.add_argument("--poll-interval", type=int, default=60, help="seconds between polls when --wait is set")
    p_download.add_argument("--mapping-csv", default=None,
                             help="path to the {batch_tag}_mapping.csv from prepare_feature_selector_batch.py -- "
                                  "if given, also writes {tag}_final_mapping.csv with selected_features/"
                                  "selection_rationale filled for every row (including debut/recent_single_point)")
    
    p_retry = sub.add_parser("retry", help="rebuild and resubmit a batch's failed requests as a new batch")
    p_retry.add_argument("--tag", required=True, help="tag of the batch whose failures should be retried")
    p_retry.add_argument("--output-dir", default=".", help="directory where {tag}_errors.csv / {tag}.csv already live")
    p_retry.add_argument("--retry-tag", default=None, help="tag for the new retry batch (default: '{tag}_retry')")
    p_retry.add_argument("--description", default="", help="optional description stored in the new batch's metadata")
    args = parser.parse_args()

    if args.command == "submit":
        submit_batch(args.jsonl, args.tag, description=args.description)
    elif args.command == "status":
        get_status(args.tag)
    elif args.command == "cancel":
        cancel_batch(args.tag)
    elif args.command == "download":
        download_and_parse(args.tag, output_dir=args.output_dir, wait=args.wait,
                            poll_interval_s=args.poll_interval, mapping_csv_path=args.mapping_csv)
    elif args.command == "retry":
        retry_failed(args.tag, output_dir=args.output_dir, retry_tag=args.retry_tag,
                     description=args.description)

if __name__ == "__main__":
    main()