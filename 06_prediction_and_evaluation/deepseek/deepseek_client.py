"""
Shared DeepSeek Chat Completions API client, used by both
s21_inference_deepseek_amazon.py and s21_inference_deepseek_douban.py.

DeepSeek's API (https://api-docs.deepseek.com/api/create-chat-completion)
differs from the vLLM-served Llama-2/Qwen2/Llama-3 backends in ways that
matter for this pipeline:

  1. NO BATCH API. Confirmed from the docs -- there's no equivalent to a
     batch endpoint, so every request goes out one at a time, sequentially.
     This matches the pipeline's existing resume/repair architecture,
     which already sends requests one at a time to vLLM too --
     run_via_deepseek below is the same shape as run_via_vllm, just
     pointed at a hosted API instead of a local server.

  2. NO `seed` PARAMETER. Not documented anywhere in the API reference.
     This is fine methodologically, not a gap to work around: a hosted
     API naturally varies run-to-run without an explicit seed, the same
     way GPT-4o-mini already does in this study -- it's only a *local*
     vLLM server that reuses the same RNG state across independent
     launches and needs an explicit --seed to get genuine variance. So
     DeepSeek replicate runs (run_1/run_2/run_3) need no seed parameter
     at all; folder-per-run is still used purely to keep results
     organized, not to force variance.

  3. NO `logit_bias`. Not documented. The emoji-banning trick used for
     Llama-2/Qwen2 (build_emoji_logit_bias) has no equivalent here and is
     dropped entirely rather than risk sending an unsupported field.

  4. `response_format` ONLY supports `{"type": "json_object"}` (generic
     JSON mode) -- NOT the strict `json_schema` mode vLLM/OpenAI support.
     The docs note json_object mode "requires explicit JSON instruction
     in prompts" -- the shared SYSTEM_PROMPT/PRODUCT_SYSTEM_PROMPT texts
     already explicitly instruct "Output a single, valid JSON object...",
     so this requirement is already satisfied without changing those
     prompts. Because there's no schema enforcement, expect a higher
     invalid-JSON rate than vLLM's json_schema mode -- the existing
     is_valid_result() + repair-pass architecture (unchanged in spirit)
     is exactly the safety net for this.

  5. No `repetition_penalty` in DeepSeek's API (that's a vLLM-specific
     extension, not part of the OpenAI-compatible spec DeepSeek follows).
     The repair pass's retry escalation uses ONLY increasing temperature,
     not the temperature+repetition_penalty combo the vLLM scripts use.

  6. No local tokenizer. DeepSeek's context window is enormous (up to
     384K tokens) relative to this pipeline's prompts (at most a few
     thousand tokens), so max_tokens is set to a fixed generous ceiling
     rather than computed from a precise prompt-token count -- there's no
     realistic way to run out of context budget here.

Model: deepseek-flash (chosen for cost/speed, comparable in scale to the
local 7B models rather than deepseek-v4-pro's flagship pricing/latency).
Thinking mode: disabled (keeps behavior directly comparable to the other
models' single 'reasoning' JSON field, instead of a separate
'reasoning_content' field with extra cost/latency).

Auth: reads DEEPSEEK_API_KEY from the environment (or --api-key). Get a
key at https://platform.deepseek.com/api_keys -- never hardcode it here.
"""
import csv
import json
import os
import time

import pandas as pd
import requests as http

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"

# DeepSeek's real context window is huge (up to 384K tokens) relative to
# this pipeline's prompts -- no local tokenizer / precise prompt-token
# budgeting needed, just cap the completion length generously (same
# ceiling the vLLM scripts use for comparability).
# MAX_TOKENS_CEILING = 1600
MAX_TOKENS_CEILING = 4096 # match with gpt4o mini's setup
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_api_key(cli_arg: str = None) -> str:
    """Resolves the DeepSeek API key from --api-key if given, else the
    DEEPSEEK_API_KEY environment variable. Fails loudly rather than
    silently sending unauthenticated requests."""
    api_key = cli_arg or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: no DeepSeek API key found -- pass --api-key or set the "
            "DEEPSEEK_API_KEY environment variable. Get a key at "
            "https://platform.deepseek.com/api_keys"
        )
    return api_key


def call_deepseek_once(body: dict, api_key: str, timeout: int = 300,
                        max_retries: int = 5, backoff_base: float = 2.0):
    """POSTs one request to the DeepSeek chat completions endpoint, with
    retry/backoff on transient errors (429 rate-limit, 5xx). A
    non-retryable error (e.g. 400 bad request) returns immediately rather
    than wasting retries on something that will never succeed.

    Returns (status_code, content, error) -- content is the assistant
    message string on success (None otherwise); error is the raw response
    text / exception string on failure (None on success)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_status, last_error = None, None

    for attempt in range(1, max_retries + 1):
        try:
            resp = http.post(DEEPSEEK_API_URL, headers=headers, json=body, timeout=timeout)
        except Exception as e:
            last_status, last_error = None, str(e)
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)
            continue

        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return resp.status_code, content, None

        last_status, last_error = resp.status_code, resp.text
        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            print(f"    DeepSeek returned {resp.status_code}, retrying (attempt {attempt}/{max_retries})...")
            time.sleep(backoff_base ** attempt)
            continue
        else:
            break  # non-retryable, or out of retries

    return last_status, None, last_error


def build_deepseek_body(system_prompt: str, user_content: str, max_tokens: int = MAX_TOKENS_CEILING,
                         temperature: float = 0.1, top_p: float = 1.0) -> dict:
    """Builds a DeepSeek chat-completions request body. Deliberately omits
    logit_bias, seed, and repetition_penalty -- none are supported by this
    API (see module docstring)."""
    return {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }


def run_via_deepseek(batch_requests, variant_tag, output_dir, api_key, sleep_seconds: float = 0.0):
    """One-by-one sequential equivalent of the vLLM scripts' run_via_vllm --
    DeepSeek has no batch API, so this sends each request individually,
    with the same incremental-save + resume behavior (skips custom_ids
    already present in an existing output file, so a re-run after a
    crash/interrupt resumes instead of starting over)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/test_pred_results_{variant_tag}.csv"

    done_ids = set()
    file_exists = os.path.isfile(out_path)
    if file_exists:
        existing = pd.read_csv(out_path)
        done_ids = set(existing["custom_id"])
        print(f"Resuming: {len(done_ids)} rows already completed, skipping those.")

    fieldnames = ["custom_id", "status_code", "content", "error"]
    n_ok, n_err, n_skipped = 0, 0, 0

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, req in enumerate(batch_requests, 1):
            if req["custom_id"] in done_ids:
                n_skipped += 1
                continue

            status_code, content, error = call_deepseek_once(req["body"], api_key)
            if content is not None:
                row = {"custom_id": req["custom_id"], "status_code": status_code, "content": content, "error": None}
                n_ok += 1
            else:
                row = {"custom_id": req["custom_id"], "status_code": status_code, "content": None, "error": error}
                n_err += 1

            writer.writerow(row)
            f.flush()  # force it to disk immediately, don't wait for buffer

            if sleep_seconds:
                time.sleep(sleep_seconds)

            if i % 10 == 0 or i == len(batch_requests):
                print(f"  [{variant_tag}] {i}/{len(batch_requests)} done (ok={n_ok}, err={n_err}, skipped={n_skipped})")

    print(f"Saved results -> {out_path} (ok={n_ok}, err={n_err}, skipped={n_skipped})")
    return out_path


def repair_incomplete_rows_deepseek(out_path: str, batch_requests_by_id: dict, is_valid_result_fn,
                                     api_key: str, max_retries: int = 3):
    """DeepSeek equivalent of the vLLM scripts' repair_incomplete_rows --
    same was_repaired/repair_temperature audit-trail behavior, but
    escalates ONLY temperature on retry (no repetition_penalty -- not
    supported by this API)."""
    df = pd.read_csv(out_path)

    if "was_repaired" not in df.columns:
        df["was_repaired"] = False
    if "repair_temperature" not in df.columns:
        df["repair_temperature"] = None

    bad_mask = ~df["content"].apply(is_valid_result_fn)
    bad_ids = df.loc[bad_mask, "custom_id"].tolist()
    print(f"{len(bad_ids)} / {len(df)} rows need repair")

    for custom_id in bad_ids:
        if custom_id not in batch_requests_by_id:
            print(f"  skip {custom_id}: no original request found")
            continue

        req = batch_requests_by_id[custom_id]
        fixed_content = None
        fixed_status = None
        fixed_temperature = None

        for attempt in range(1, max_retries + 1):
            retry_body = json.loads(json.dumps(req["body"]))  # deep copy
            retry_body["temperature"] = min(0.2 + 0.2 * attempt, 0.8)

            status_code, content, error = call_deepseek_once(retry_body, api_key)
            if content is None:
                print(f"  {custom_id} attempt {attempt} errored: {error}")
                continue

            if is_valid_result_fn(content):
                fixed_content = content
                fixed_status = status_code
                fixed_temperature = retry_body["temperature"]
                print(f"  {custom_id} fixed on attempt {attempt}")
                break
            else:
                print(f"  {custom_id} attempt {attempt} still invalid")

        idx = df.index[df["custom_id"] == custom_id][0]
        if fixed_content is not None:
            df.at[idx, "content"] = fixed_content
            df.at[idx, "status_code"] = fixed_status
            df.at[idx, "error"] = None
            df.at[idx, "was_repaired"] = True
            df.at[idx, "repair_temperature"] = fixed_temperature
        else:
            print(f"  {custom_id} still bad after {max_retries} attempts — leaving as-is for manual review")
            df.at[idx, "was_repaired"] = True  # attempted but not fixed

    df.to_csv(out_path, index=False)
    still_bad = (~df["content"].apply(is_valid_result_fn)).sum()
    print(f"Repair pass done. {still_bad} rows remain invalid.")
