"""
OpenAI Batch API pipeline -- submit, check, and download, all in one file.
Usage:
    python s12_batch_exp_pipeline.py submit  --jsonl path/to/requests.jsonl --tag my_batch
    python s12_batch_exp_pipeline.py status  --tag my_batch
    python s12_batch_exp_pipeline.py download --tag my_batch --output-dir out/ [--wait]

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


def strip_trailing_commas(text):
    """Fix the one malformed-JSON pattern we've actually seen models emit:
    a trailing comma right before a closing '}' or ']' (valid in Python/JS,
    not valid JSON), e.g. '..."reasoning": "...",\\n}'. Only removes a comma
    that is immediately followed (ignoring whitespace) by a closing bracket
    -- never touches a comma that's actually separating two real items."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


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

    # walk back over the trailing run of closing brackets (e.g. '}' or ']}')
    i = len(text)
    while i > 0 and text[i - 1] in "}]":
        i -= 1
    tail_close = text[i:]
    body = text[:i]

    if body.endswith('"'):
        return text  # already properly closed -- nothing to repair

    # strip a trailing run of non-ASCII / stray bracket-like characters
    # (the garbage tokens actually observed), back to real prose content
    stripped = body.rstrip()
    while stripped and (ord(stripped[-1]) > 0x2000 or stripped[-1] in "()[]{}<>"):
        stripped = stripped[:-1]

    if stripped == body:
        return text  # nothing recognizable as garbage -- don't guess further

    return stripped.rstrip() + '"' + tail_close


def parse_json_safe(text):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned), None
    except (json.JSONDecodeError, TypeError) as e:
        # second attempt: trailing-comma repair
        try:
            return json.loads(strip_trailing_commas(cleaned)), None
        except (json.JSONDecodeError, TypeError):
            pass
        # third attempt: unterminated-trailing-string repair (stray garbage
        # tokens in place of a closing quote)
        try:
            return json.loads(repair_unterminated_trailing_string(cleaned)), None
        except (json.JSONDecodeError, TypeError):
            return {}, str(e)


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
# download + parse (one pass)
# ---------------------------------------------------------------------------

# def download_and_parse(tag, output_dir=".", poll_interval_s=60, wait=False, client=None):
#     client = client or _client()
#     meta = _load_meta(tag)
#     os.makedirs(output_dir, exist_ok=True)

#     batch = client.batches.retrieve(meta["batch_id"])
#     if wait:
#         while batch.status not in ("completed", "failed", "expired", "cancelled"):
#             comp = batch.request_counts.completed
#             total = batch.request_counts.total
#             print(f"  [{tag}] {batch.status} ({comp}/{total}) -- polling again in {poll_interval_s}s")
#             time.sleep(poll_interval_s)
#             batch = client.batches.retrieve(meta["batch_id"])

#     if batch.status != "completed":
#         print(f"Batch not ready (status: {batch.status}). Re-run with --wait, or check again later.")
#         return None

#     raw_jsonl_path = os.path.join(output_dir, f"{tag}_raw.jsonl")
#     csv_path = os.path.join(output_dir, f"{tag}.csv")
#     err_csv_path = os.path.join(output_dir, f"{tag}_errors.csv")

#     rows = []
#     if batch.output_file_id:
#         content = client.files.content(batch.output_file_id).content
#         with open(raw_jsonl_path, "wb") as f:
#             f.write(content)

#         n_parse_failed = 0
#         with open(raw_jsonl_path, "r") as f:
#             for line in f:
#                 data = json.loads(line)
#                 body = data.get("response", {}).get("body", {})
#                 choices = body.get("choices") or []
#                 if not choices:
#                     continue
#                 message_content = choices[0]["message"]["content"]
#                 parsed, parse_error = parse_json_safe(message_content)
#                 row = {"custom_id": data["custom_id"], **parsed}
#                 if parse_error is not None:
#                     n_parse_failed += 1
#                     row["_parse_error"] = parse_error
#                     row["_raw_content"] = message_content
#                 rows.append(row)

#         if n_parse_failed:
#             print(f"⚠️  {n_parse_failed} response(s) had malformed JSON that couldn't be "
#                   f"parsed even after trailing-comma repair -- these rows have empty "
#                   f"selected_features/reasoning but a '_parse_error' and '_raw_content' "
#                   f"column so you can inspect what the model actually returned.")

#         if rows:
#             df = pd.DataFrame(rows)
#             df.to_csv(csv_path, index=False)
#             print(f"Parsed {len(df)} responses -> {csv_path}")
#         else:
#             print("No successful responses to parse.")
#     else:
#         print("No output file (0 successful requests).")

#     if batch.error_file_id:
#         err_content = client.files.content(batch.error_file_id).content
#         errors = []
#         for line in err_content.decode("utf-8").splitlines():
#             data = json.loads(line)
#             error_info = data.get("error") or {}
#             errors.append({
#                 "custom_id": data.get("custom_id"),
#                 "error_code": error_info.get("code"),
#                 "error_message": error_info.get("message"),
#             })
#         if errors:
#             err_df = pd.DataFrame(errors)
#             err_df.to_csv(err_csv_path, index=False)
#             print(f"{len(errors)} failed requests -> {err_csv_path}")
#             print(err_df["error_code"].value_counts().to_string())
#     else:
#         print("No errors reported.")

#     return csv_path

def download_and_parse(tag, output_dir=".", poll_interval_s=60, wait=False, client=None):
    client = client or _client()
    meta = _load_meta(tag)
    os.makedirs(output_dir, exist_ok=True)

    batch = client.batches.retrieve(meta["batch_id"])
    if wait:
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            comp = batch.request_counts.completed
            total = batch.request_counts.total
            print(f"  [{tag}] {batch.status} ({comp}/{total}) -- polling again in {poll_interval_s}s")
            time.sleep(poll_interval_s)
            batch = client.batches.retrieve(meta["batch_id"])

    if batch.status != "completed":
        print(f"Batch not ready (status: {batch.status}). Re-run with --wait, or check again later.")
        return None

    raw_jsonl_path = os.path.join(output_dir, f"{tag}_raw.jsonl")
    csv_path = os.path.join(output_dir, f"{tag}.csv")
    err_csv_path = os.path.join(output_dir, f"{tag}_errors.csv")

    rows = []
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).content
        with open(raw_jsonl_path, "wb") as f:
            f.write(content)

        n_empty = 0
        with open(raw_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                body = data.get("response", {}).get("body", {})
                choices = body.get("choices") or []
                if not choices:
                    n_empty += 1
                    continue
                message_content = choices[0]["message"]["content"]
                rows.append({
                    "custom_id": data["custom_id"],
                    "explanatory_factors": message_content,
                })

        if n_empty:
            print(f"⚠️  {n_empty} response(s) had no choices returned and were skipped.")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} responses -> {csv_path}")
        else:
            print("No successful responses to save.")
    else:
        print("No output file (0 successful requests).")

    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).content
        errors = []
        for line in err_content.decode("utf-8").splitlines():
            data = json.loads(line)
            error_info = data.get("error") or {}
            errors.append({
                "custom_id": data.get("custom_id"),
                "error_code": error_info.get("code"),
                "error_message": error_info.get("message"),
            })
        if errors:
            err_df = pd.DataFrame(errors)
            err_df.to_csv(err_csv_path, index=False)
            print(f"{len(errors)} failed requests -> {err_csv_path}")
            print(err_df["error_code"].value_counts().to_string())
    else:
        print("No errors reported.")

    return csv_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API pipeline: submit / status / download")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="upload a JSONL file and create a batch job")
    p_submit.add_argument("--jsonl", required=True, help="path to the batch input JSONL")
    p_submit.add_argument("--tag", required=True, help="short identifier for this batch (used to name the metadata/output files)")
    p_submit.add_argument("--description", default="", help="optional description stored in batch metadata")

    p_status = sub.add_parser("status", help="check a batch's current status")
    p_status.add_argument("--tag", required=True)

    p_download = sub.add_parser("download", help="download and parse a completed batch's results")
    p_download.add_argument("--tag", required=True)
    p_download.add_argument("--output-dir", default=".", help="directory to save the parsed CSV / raw JSONL / errors into")
    p_download.add_argument("--wait", action="store_true", help="poll until the batch completes, then download automatically")
    p_download.add_argument("--poll-interval", type=int, default=60, help="seconds between polls when --wait is set")

    args = parser.parse_args()

    if args.command == "submit":
        submit_batch(args.jsonl, args.tag, description=args.description)
    elif args.command == "status":
        get_status(args.tag)
    elif args.command == "download":
        download_and_parse(args.tag, output_dir=args.output_dir, wait=args.wait, poll_interval_s=args.poll_interval)


if __name__ == "__main__":
    main()