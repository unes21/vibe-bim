"""
run_e2e_benchmark.py — VIBE End-to-End Benchmark Harness
=========================================================
Exercises the full command pipeline for every entry in TEST_CORPUS:

    (1) NLP resolution    — /api/llm/category
    (2) Intent dispatch   — /api/write  (marks elements Kilit=True)
    (3) Dynamo execution  — VIBE_executor.dyn tick (periodic)
    (4) Write verification — /api/category_status
                               checks Kilit flipped back to False,
                               LastWrite timestamp recorded,
                               LastError empty

Produces two metrics per command:
    nlp_correct    — did the NLP layer map the input to the gold category?
    e2e_verified   — did Dynamo actually write the unique note into Revit
                     and release the lock within the timeout window?

Prerequisites
-------------
* Autodesk Revit is open with racbasicsampleproject.rvt loaded
* Dynamo Player is running VIBE_executor.dyn in periodic mode
* Flask server (app.py) is running and reachable at --server
* revit_data.json has been pre-populated with element_type fields by
  whichever Dynamo graph the project uses for model ingest

Usage
-----
    python run_e2e_benchmark.py                         # all defaults
    python run_e2e_benchmark.py --wait-sec 25           # slower Dynamo tick
    python run_e2e_benchmark.py --subset wall,floor     # test only 2 categories
    python run_e2e_benchmark.py --dry-run               # NLP only, skip write

Author  : Ayberk Enis
Project : VIBE
License : MIT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

# Reuse the 91-command corpus from run_benchmark.py (side-by-side file)
try:
    from run_benchmark import TEST_CORPUS
except ImportError:
    sys.exit("run_benchmark.py must live next to this file (TEST_CORPUS is imported).")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def resolve_category(server: str, text: str, timeout: float) -> Dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{server}/api/llm/category",
                          json={"text": text}, timeout=timeout)
        data = r.json() if r.status_code == 200 else {}
    except Exception as exc:
        data = {"category": None, "source": "http_error", "error": str(exc)[:200]}
    data.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))
    return data


def dispatch_write(server: str, cat: str, note: str, mode: str,
                   timeout: float) -> Dict:
    try:
        r = requests.post(f"{server}/api/write",
                          json={"cat": cat, "note": note, "mode": mode},
                          timeout=timeout)
        return r.json() if r.status_code == 200 else {
            "msg": f"HTTP {r.status_code}", "count": 0,
        }
    except Exception as exc:
        return {"msg": f"http_error: {exc}"[:200], "count": 0}


def poll_status(server: str, cat: str, note_substr: str,
                timeout: float) -> Dict:
    try:
        r = requests.post(f"{server}/api/category_status",
                          json={"cat": cat, "note_substr": note_substr},
                          timeout=timeout)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Verification loop
# ---------------------------------------------------------------------------

def wait_for_dynamo_write(
    server: str,
    cat: str,
    note_substr: str,
    max_wait_sec: float,
    poll_interval: float,
    timeout: float,
) -> Tuple[bool, float, Dict]:
    """
    Poll /api/category_status until at least one element matching *cat* has
    our *note_substr* in Asistan_Notu AND has Kilit==False (Dynamo released
    the lock) AND has a LastWrite timestamp AND has no LastError.

    Returns (verified, elapsed_seconds, last_status_snapshot).
    """
    t0 = time.time()
    last: Dict = {}
    while time.time() - t0 < max_wait_sec:
        last = poll_status(server, cat, note_substr, timeout)
        if last:
            with_note       = last.get("with_note", 0) or 0
            still_locked    = last.get("still_locked", 0) or 0
            with_last_write = last.get("with_last_write", 0) or 0
            with_error      = last.get("with_error", 0) or 0

            # Success criterion: at least one element was tagged with our
            # note, Dynamo cleared its lock, a LastWrite was recorded, and
            # no error surfaced on that element.
            verified_count = max(0, with_note - still_locked - with_error)
            wrote_count    = min(with_last_write, verified_count)
            if wrote_count > 0:
                return True, time.time() - t0, last
        time.sleep(poll_interval)
    return False, time.time() - t0, last


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_one_command(
    server: str,
    gold: str, lang: str, variant: str, text: str,
    *,
    max_wait_sec: float,
    poll_interval: float,
    http_timeout: float,
    dry_run: bool,
) -> Dict:
    # 1. NLP resolution
    nlp = resolve_category(server, text, http_timeout)
    resolved = nlp.get("category")
    source   = nlp.get("source")
    nlp_ms   = nlp.get("latency_ms")
    nlp_correct = (resolved == gold)

    row: Dict = {
        "gold": gold, "lang": lang, "variant": variant, "input": text,
        "nlp_resolved": resolved, "nlp_source": source,
        "nlp_latency_ms": nlp_ms, "nlp_correct": nlp_correct,
        "e2e_verified": None, "e2e_elapsed_sec": None,
        "dispatched_count": 0, "write_source_cleanup_note": "",
        "last_status": None,
    }

    if dry_run:
        row["write_source_cleanup_note"] = "dry_run"
        return row

    # Cannot dispatch a write if NLP didn't resolve anything
    if resolved is None:
        row["write_source_cleanup_note"] = "no_category_resolved"
        return row

    # 2. Dispatch with a unique note — lets verification distinguish this
    #    command's write from any prior state.
    unique_tag = f"VIBE-E2E-{uuid.uuid4().hex[:8]}"
    note       = f"{unique_tag} :: {text[:40]}"
    write_resp = dispatch_write(server, resolved, note, mode="overwrite",
                                timeout=http_timeout)
    row["dispatched_count"] = write_resp.get("count", 0) or 0

    if row["dispatched_count"] == 0:
        # Category resolved but no matching elements in the model.
        row["e2e_verified"] = False
        row["write_source_cleanup_note"] = "no_elements_of_category_in_model"
        return row

    # 3. Wait for Dynamo to tick and execute the write
    verified, elapsed, last_status = wait_for_dynamo_write(
        server=server, cat=resolved, note_substr=unique_tag,
        max_wait_sec=max_wait_sec, poll_interval=poll_interval,
        timeout=http_timeout,
    )
    row["e2e_verified"]    = verified
    row["e2e_elapsed_sec"] = round(elapsed, 2)
    row["last_status"]     = last_status
    return row


def run_benchmark(
    server: str,
    out_csv: str,
    out_summary: str,
    subset: List[str],
    max_wait_sec: float,
    poll_interval: float,
    http_timeout: float,
    inter_cmd_sleep: float,
    dry_run: bool,
) -> None:
    corpus = [row for row in TEST_CORPUS if (not subset or row[0] in subset)]
    total = len(corpus)
    if total == 0:
        sys.exit("Empty corpus after subset filter — check --subset.")
    print(f"Running {total} commands "
          f"(dry_run={dry_run}, max_wait={max_wait_sec}s)\n")

    rows: List[Dict] = []
    for idx, (gold, lang, variant, text) in enumerate(corpus, 1):
        print(f"[{idx:3d}/{total}] {gold:10s} / {lang} / {variant:12s}  :: {text}")
        row = run_one_command(
            server=server, gold=gold, lang=lang, variant=variant, text=text,
            max_wait_sec=max_wait_sec, poll_interval=poll_interval,
            http_timeout=http_timeout, dry_run=dry_run,
        )
        rows.append(row)

        nlp_mark = "✓" if row["nlp_correct"] else "✗"
        e2e_mark = (
            "—" if row["e2e_verified"] is None else
            "✓" if row["e2e_verified"] else
            "✗"
        )
        print(f"          NLP {nlp_mark} ({row['nlp_source']}, "
              f"{row['nlp_latency_ms']} ms)   "
              f"E2E {e2e_mark} (n={row['dispatched_count']}, "
              f"{row['e2e_elapsed_sec']}s)")

        if inter_cmd_sleep > 0:
            time.sleep(inter_cmd_sleep)

    write_csv(rows, out_csv)
    write_summary(rows, out_summary, server)


def write_csv(rows: List[Dict], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gold", "lang", "variant", "input",
            "nlp_resolved", "nlp_source", "nlp_latency_ms", "nlp_correct",
            "e2e_verified", "e2e_elapsed_sec",
            "dispatched_count", "write_source_cleanup_note",
        ])
        writer.writeheader()
        for r in rows:
            slim = {k: r.get(k) for k in writer.fieldnames}
            writer.writerow(slim)


def write_summary(rows: List[Dict], out_summary: str, server: str) -> None:
    with open(out_summary, "w", encoding="utf-8") as f:
        def w(s=""):
            print(s)
            f.write(s + "\n")

        N = len(rows)
        nlp_ok = sum(1 for r in rows if r["nlp_correct"])
        e2e_attempted = [r for r in rows if r["e2e_verified"] is not None]
        e2e_ok = sum(1 for r in e2e_attempted if r["e2e_verified"])

        by_src = Counter(r["nlp_source"] for r in rows)

        w("=" * 72)
        w(f"VIBE End-to-End Benchmark Summary — "
          f"{datetime.now().isoformat(timespec='seconds')}")
        w(f"Server: {server}")
        w(f"N = {N} commands")
        w(f"NLP accuracy : {nlp_ok}/{N} = {100 * nlp_ok / N:.1f}%")
        if e2e_attempted:
            pct = 100 * e2e_ok / len(e2e_attempted)
            w(f"E2E verified : {e2e_ok}/{len(e2e_attempted)} = {pct:.1f}%  "
              f"(of commands that reached dispatch)")
        w("=" * 72)

        w("\n-- NLP tier distribution --")
        for src, cnt in by_src.most_common():
            w(f"  {str(src):16s} {cnt:3d}  ({100 * cnt / N:.1f}%)")

        # per-category breakdown
        cats = sorted({r["gold"] for r in rows})
        w("\n-- Per-category NLP accuracy × E2E verification --")
        w(f"  {'cat':10s}  {'N':>3s}  {'NLP ok':>7s}  {'E2E ok':>7s}  "
          f"{'mean E2E sec':>14s}  {'dispatch misses':>16s}")
        for cat in cats:
            cat_rows = [r for r in rows if r["gold"] == cat]
            n        = len(cat_rows)
            nok      = sum(1 for r in cat_rows if r["nlp_correct"])
            eok      = sum(1 for r in cat_rows if r["e2e_verified"])
            elapsed  = [r["e2e_elapsed_sec"] for r in cat_rows
                        if isinstance(r["e2e_elapsed_sec"], (int, float))
                        and r["e2e_verified"]]
            mean_e   = statistics.mean(elapsed) if elapsed else 0.0
            dmiss    = sum(1 for r in cat_rows
                           if r["e2e_verified"] is False
                           and r["dispatched_count"] == 0)
            w(f"  {cat:10s}  {n:3d}  "
              f"{nok}/{n:<3d}   {eok}/{n:<3d}   "
              f"{mean_e:>14.2f}  {dmiss:>16d}")

        # latency stats per NLP tier
        w("\n-- NLP latency per tier (ms) --")
        lat = defaultdict(list)
        for r in rows:
            if isinstance(r["nlp_latency_ms"], (int, float)):
                lat[r["nlp_source"]].append(float(r["nlp_latency_ms"]))
        w(f"  {'tier':16s} {'n':>4s} {'min':>8s} {'median':>8s} "
          f"{'mean':>8s} {'max':>8s}")
        for src in sorted(lat):
            vals = lat[src]
            if vals:
                w(f"  {src:16s} {len(vals):4d} {min(vals):8.1f} "
                  f"{statistics.median(vals):8.1f} "
                  f"{statistics.mean(vals):8.1f} {max(vals):8.1f}")

        # e2e elapsed stats
        e2e_elapsed = [r["e2e_elapsed_sec"] for r in rows
                       if isinstance(r["e2e_elapsed_sec"], (int, float))
                       and r["e2e_verified"]]
        if e2e_elapsed:
            w("\n-- Dynamo tick + write verification elapsed time (seconds) --")
            w(f"  n={len(e2e_elapsed)}   min={min(e2e_elapsed):.2f}   "
              f"median={statistics.median(e2e_elapsed):.2f}   "
              f"mean={statistics.mean(e2e_elapsed):.2f}   "
              f"max={max(e2e_elapsed):.2f}")

        w("\nOutputs:")
        w(f"  per-command CSV : {os.path.abspath(out_summary).replace('.txt', '.csv')}")
        w(f"  summary         : {os.path.abspath(out_summary)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="VIBE end-to-end benchmark harness")
    ap.add_argument("--server",  default="http://127.0.0.1:5000")
    ap.add_argument("--out-csv", default="vibe_e2e_results.csv")
    ap.add_argument("--out-summary", default="vibe_e2e_summary.txt")
    ap.add_argument("--subset", default="",
                    help="Comma-separated category keys to restrict the run "
                         "(e.g. wall,floor). Empty = all 13 categories.")
    ap.add_argument("--wait-sec", type=float, default=20.0,
                    help="Max seconds to wait for Dynamo write per command.")
    ap.add_argument("--poll-sec", type=float, default=0.5,
                    help="Polling interval while waiting for Dynamo tick.")
    ap.add_argument("--http-timeout", type=float, default=45.0,
                    help="HTTP request timeout for /api/* calls.")
    ap.add_argument("--sleep-between", type=float, default=0.5,
                    help="Pause between commands (helps Dynamo settle).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip /api/write and verification; NLP-only pass.")
    args = ap.parse_args()

    subset = [s.strip() for s in args.subset.split(",") if s.strip()]

    # Probe server health once, print and abort if unreachable
    try:
        h = requests.get(f"{args.server}/api/health", timeout=5).json()
    except Exception as exc:
        sys.exit(f"Could not reach {args.server}/api/health: {exc}\n"
                 f"Is 'python app.py' running?")
    print("Server health:")
    print(json.dumps(h, indent=2, ensure_ascii=False))

    if not args.dry_run and not h.get("json_exists"):
        print("\n[WARN] revit_data.json does not exist at the server path. "
              "Dispatch will succeed but Dynamo has nothing to write. "
              "Ensure Revit + Dynamo ingest graph have populated it first.\n")

    print()
    run_benchmark(
        server=args.server,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        subset=subset,
        max_wait_sec=args.wait_sec,
        poll_interval=args.poll_sec,
        http_timeout=args.http_timeout,
        inter_cmd_sleep=args.sleep_between,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
