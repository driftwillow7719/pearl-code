#!/usr/bin/env python3
"""
retrieval_quality.py
Post-hoc retrieval-quality metrics (no LLM calls) for the Retrieval_Quality tab.

Metrics (per the sheet definitions):
  direction   : per query, mean indicator[sign(case) == sign(query)] over valid cases,
                then average EQUALLY over target classes increase / stable / decrease.
  log_change  : g = ln(1+next) - ln(1+latest); per query mean |g_q - g_c|; mean over queries.
  log_pop     : per query mean |ln(1+next_q) - ln(1+next_c)|; mean over queries.
  content_alignment : (optional, needs --query-embeddings/--pool-embeddings) per query mean
                cosine similarity between the query's embedding and each retrieved case's
                embedding. Independent of outcome/ground-truth -- measures whether retrieved
                items are actually close to the QUERY item itself (not the cluster center),
                as opposed to direction/log_change/log_pop which measure outcome alignment.
Debut queries (unknown latest) are excluded from direction / log_change (baseline is NOT
filled with zero) and reported separately for log_pop + coverage.

INPUT FILES
  --queries CSV : one row per query. Columns (names configurable):
        --q-id-col (default doc_id), --q-latest-col (default latest_pop_value; NaN/empty = debut),
        --q-next-col (default next_pop_value, ground truth, unclipped),
        optional --q-tier-col (training-defined popularity tier).
  --pool CSV    : retrieval corpus (training pool). Columns:
        --p-id-col (default doc_id), --p-latest-col, --p-next-col  (same semantics).
  --variant NAME=PATH : repeatable. PATH is the retrieval-context CSV of one experimental
        variant; needs --r-id-col (default doc_id = query id) and --r-ids-col
        (default retrieved_ids = ranked list of pool ids; JSON list, or separated by ; | , space).
  --k : metrics use first k retrieved ids (use the k picked on validation).
  --query-embeddings PATH (optional) : .npy array of query embeddings. Row i corresponds to
        row i of --queries CSV, in the CSV's ORIGINAL file order (i.e. before any row-dropping
        by --filter-ids-csv -- alignment is resolved against the raw read, not the filtered one).
  --pool-embeddings PATH (optional) : .npy array of pool embeddings. Row i corresponds to row i
        of --pool CSV (same positional-index convention).
        If either embeddings arg is omitted, content_alignment is just left as NaN everywhere
        -- nothing else changes.

OUTPUT
  <out-prefix>_summary.csv   one row per variant (+ breakdown rows)
  <out-prefix>_per_query.csv per query per variant (for debugging / re-aggregation)
  Prints sanity check: Full vs selected-group-random direction agreement must be identical
  on matched queries/pairs (use --check-pair FULL_NAME,RANDOMCASES_NAME).
"""
import argparse, json, re, sys
import numpy as np
import pandas as pd


def parse_ids(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            s = s.strip("[]")
    return [t for t in re.split(r"[;|,\s]+", s.replace("'", "").replace('"', "")) if t]


def sgn(x, eps):
    return 0 if abs(x) <= eps else (1 if x > 0 else -1)


def cls_name(s):
    return {1: "increase", 0: "stable", -1: "decrease"}[s]


def cosine_sim(a, b):
    """a, b: 1-D np.array or None. Returns float cosine similarity, or NaN if either is
    missing or has zero norm (e.g. placeholder embedding)."""
    if a is None or b is None:
        return np.nan
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def per_query_metrics(qrow, case_rows, eps):
    """qrow: dict(latest,next,embedding); case_rows: list of dict(latest,next,embedding).
    Returns dict."""
    q_next = qrow["next"]
    q_latest = qrow["latest"]
    q_known = not np.isnan(q_latest)
    out = {"n_retrieved": len(case_rows), "q_debut": not q_known}
    # log-pop gap: needs only outcomes
    lp = [abs(np.log1p(q_next) - np.log1p(c["next"])) for c in case_rows if not np.isnan(c["next"])]
    out["log_pop"] = float(np.mean(lp)) if lp else np.nan
    out["n_logpop_valid"] = len(lp)
    # content alignment: cosine similarity to the QUERY item itself (outcome-independent --
    # computed regardless of whether ground truth / debut status is known)
    q_emb = qrow.get("embedding")
    sims = [cosine_sim(q_emb, c.get("embedding")) for c in case_rows]
    sims = [s for s in sims if not np.isnan(s)]
    out["content_alignment"] = float(np.mean(sims)) if sims else np.nan
    out["n_content_valid"] = len(sims)
    # direction / change: need both baselines known
    if q_known:
        gq = np.log1p(q_next) - np.log1p(q_latest)
        sq = sgn(q_next - q_latest, eps)
        valid = [c for c in case_rows if not np.isnan(c["latest"]) and not np.isnan(c["next"])]
        out["n_dir_valid"] = len(valid)
        if valid:
            ind = [float(sgn(c["next"] - c["latest"], eps) == sq) for c in valid]
            gc = [abs(gq - (np.log1p(c["next"]) - np.log1p(c["latest"]))) for c in valid]
            out["direction"] = float(np.mean(ind))
            out["log_change"] = float(np.mean(gc))
        else:
            out["direction"] = np.nan
            out["log_change"] = np.nan
        out["q_class"] = cls_name(sq)
    else:
        out.update({"n_dir_valid": 0, "direction": np.nan, "log_change": np.nan, "q_class": None})
    return out


def aggregate(pq, label):
    """pq: per-query dataframe (one variant). Returns dict of summary stats."""
    obs = pq[~pq.q_debut]
    deb = pq[pq.q_debut]
    cls_means = []
    for c in ["increase", "stable", "decrease"]:
        v = obs.loc[obs.q_class == c, "direction"].dropna()
        if len(v):
            cls_means.append(v.mean())
    return {
        "subset": label,
        "n_queries": len(pq),
        "n_observed_baseline": len(obs),
        "n_debut": len(deb),
        "coverage_direction": float((obs.n_dir_valid > 0).mean()) if len(obs) else np.nan,
        "direction_balanced": float(np.mean(cls_means)) if cls_means else np.nan,
        "log_change_gap": float(obs.log_change.dropna().mean()) if obs.log_change.notna().any() else np.nan,
        "log_pop_gap_observed": float(obs.log_pop.dropna().mean()) if obs.log_pop.notna().any() else np.nan,
        "log_pop_gap_debut": float(deb.log_pop.dropna().mean()) if deb.log_pop.notna().any() else np.nan,
        "log_pop_gap_all": float(pq.log_pop.dropna().mean()) if pq.log_pop.notna().any() else np.nan,
        "content_alignment": float(pq.content_alignment.dropna().mean()) if pq.content_alignment.notna().any() else np.nan,
        "coverage_content": float((pq.n_content_valid > 0).mean()) if len(pq) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--filter-ids-csv", default=None, help="CSV list of items to be kept mis. 3603 item TRAIL")
    ap.add_argument("--filter-col", default="movie_id", help="column name")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--q-id-col", default="doc_id")
    ap.add_argument("--q-latest-col", default="latest_pop_value")
    ap.add_argument("--q-next-col", default="next_pop_value")
    ap.add_argument("--q-tier-col", default=None)
    ap.add_argument("--p-id-col", default="doc_id")
    ap.add_argument("--p-latest-col", default="latest_pop_value")
    ap.add_argument("--p-next-col", default="next_pop_value")
    ap.add_argument("--r-id-col", default="doc_id")
    ap.add_argument("--r-ids-col", default="retrieved_ids")
    ap.add_argument("--stable-eps", type=float, default=0.0,
                    help="|next-latest|<=eps counts as 'stable' (default 0 = exact sign)")
    ap.add_argument("--check-pair", default=None,
                    help="FULL_NAME,SELECTED_GROUP_RANDOM_NAME for the identical-direction check")
    ap.add_argument("--query-embeddings", default=None,
                    help=".npy array, row i aligned to row i of --queries CSV's RAW file order "
                         "(before --filter-ids-csv row-dropping). Omit to skip content_alignment.")
    ap.add_argument("--pool-embeddings", default=None,
                    help=".npy array, row i aligned to row i of --pool CSV's file order. "
                         "Omit to skip content_alignment.")
    args = ap.parse_args()

    qdf = pd.read_csv(args.queries, dtype={args.q_id_col: str})

    query_emb_full = None
    if args.query_embeddings:
        query_emb_full = np.load(args.query_embeddings)
        if len(query_emb_full) != len(qdf):
            sys.exit(f"[error] --query-embeddings has {len(query_emb_full)} rows but "
                      f"--queries CSV has {len(qdf)} rows (raw, pre-filter) -- row-count "
                      f"mismatch, alignment would be wrong. Check the embeddings file matches "
                      f"this exact queries CSV.")

    # filter according to TRAIL set -- NOTE: this keeps qdf's original row index (no
    # reset_index), so query_emb_full[idx] below still points at the right embedding row
    # for whichever rows survive the filter.
    if args.filter_ids_csv:
        keep_ids = set(pd.read_csv(args.filter_ids_csv)[args.filter_col])
        n0 = len(qdf)
        qdf = qdf[qdf[args.filter_col].isin(keep_ids)].copy()
        print(f"[filter] queries {n0} -> {len(qdf)} by {args.filter_col}; "
            f"filter IDs not found in queries: {len(keep_ids - set(qdf[args.filter_col]))}")

    pdf_ = pd.read_csv(args.pool, dtype={args.p_id_col: str})

    pool_emb_full = None
    if args.pool_embeddings:
        pool_emb_full = np.load(args.pool_embeddings)
        if len(pool_emb_full) != len(pdf_):
            sys.exit(f"[error] --pool-embeddings has {len(pool_emb_full)} rows but "
                      f"--pool CSV has {len(pdf_)} rows -- row-count mismatch, alignment "
                      f"would be wrong. Check the embeddings file matches this exact pool CSV.")

    q = {}
    for idx, r in qdf.iterrows():
        q[str(r[args.q_id_col])] = {
            "latest": pd.to_numeric(r[args.q_latest_col], errors="coerce"),
            "next": pd.to_numeric(r[args.q_next_col], errors="coerce"),
            "tier": r[args.q_tier_col] if args.q_tier_col else None,
            "embedding": query_emb_full[idx] if query_emb_full is not None else None,
        }
    pool = {}
    for idx, r in pdf_.iterrows():
        pool[str(r[args.p_id_col])] = {
            "latest": pd.to_numeric(r[args.p_latest_col], errors="coerce"),
            "next": pd.to_numeric(r[args.p_next_col], errors="coerce"),
            "embedding": pool_emb_full[idx] if pool_emb_full is not None else None,
        }

    summaries, pq_all = [], []
    per_variant = {}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        rdf = pd.read_csv(path, dtype={args.r_id_col: str})
        rows = []
        missing_cases = 0
        if rdf.empty:
            continue
        for _, r in rdf.iterrows():
            qid = str(r[args.r_id_col])
            if qid not in q or np.isnan(q[qid]["next"]):
                continue  # no ground truth -> not scored (never convert to zero)
            ids = [str(i) for i in parse_ids(r[args.r_ids_col])[: args.k]]
            cases = []
            for cid in ids:
                if cid in pool:
                    cases.append(pool[cid])
                else:
                    missing_cases += 1  # unknown case record: skip, don't zero-fill
            m = per_query_metrics(q[qid], cases, args.stable_eps)
            m.update({"variant": name, "query_id": qid, "tier": q[qid]["tier"],
                      "q_next": q[qid]["next"], "q_positive": q[qid]["next"] > 0})
            rows.append(m)
        pq = pd.DataFrame(rows)
        if pq.empty:
            sys.exit(f"[error] {name}: no query matched the queries file (check id columns / id formats)")
        per_variant[name] = pq
        pq_all.append(pq)
        if missing_cases:
            print(f"[warn] {name}: {missing_cases} retrieved ids not found in pool (skipped)")
        s = aggregate(pq, "all"); s["variant"] = name; summaries.append(s)
        s = aggregate(pq[pq.q_positive], "positive_outcome"); s["variant"] = name; summaries.append(s)
        if args.q_tier_col:
            for t, g in pq.groupby("tier"):
                s = aggregate(g, f"tier={t}"); s["variant"] = name; summaries.append(s)

    summ = pd.DataFrame(summaries)
    cols = ["variant", "subset"] + [c for c in summ.columns if c not in ("variant", "subset")]
    summ = summ[cols]
    summ.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    pd.concat(pq_all).to_csv(f"{args.out_prefix}_per_query.csv", index=False)
    print(summ[summ.subset == "all"].to_string(index=False))

    if args.check_pair:
        a, b = args.check_pair.split(",")
        A = per_variant[a].set_index("query_id")
        B = per_variant[b].set_index("query_id")
        common = A.index.intersection(B.index)
        ok = (A.loc[common, "n_dir_valid"] == B.loc[common, "n_dir_valid"])
        both = common[ok.values]
        diff = (A.loc[both, "direction"] - B.loc[both, "direction"]).abs().max()
        print(f"\n[check] {a} vs {b}: {len(both)}/{len(common)} queries with equal valid-case counts; "
              f"max |direction diff| = {diff}")
        if len(both) < len(common):
            print("  -> valid pair sets differ for some queries; restrict to matched pairs before comparing.")
        if pd.notna(diff) and diff > 1e-9:
            print("  !! direction should be identical when all cases come from the same group; check routing/inputs")


if __name__ == "__main__":
    main()