"""
aggregate_runs.py - Aggregate N benchmark CSVs and report mean +/- std
======================================================================
Reads the per-run CSVs produced by run_multi.py (or any directory of
vibe_e2e_results.csv files) and produces:

    * headline metrics with mean +/- std across runs
        - NLP accuracy
        - dispatch rate
        - write success on dispatched subset
        - strict end-to-end accuracy (NLP correct AND write succeeded)
    * per-category mean +/- std CSV
    * per-tier breakdown (rules / ollama / unresolved) with std
    * per-variant breakdown (a..g) with std
    * per-command flakiness report: which commands flip nlp_correct or
      e2e_verified across runs

Usage
-----
    # Use a manifest from run_multi.py
    python aggregate_runs.py --manifest runs/multirun_manifest.json

    # Or point at a directory of CSVs directly
    python aggregate_runs.py --csv-dir runs/

    # Or pass explicit files
    python aggregate_runs.py --csvs run01.csv run02.csv run03.csv

"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _to_bool(s) -> bool:
    """CSV columns can be 'True'/'False'/'' — coerce to bool."""
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() == "true"


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def load_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "gold":              r.get("gold", "").strip(),
                "lang":              r.get("lang", "").strip(),
                "variant":           r.get("variant", "").strip(),
                "input":             r.get("input", "").strip(),
                "nlp_resolved":      (r.get("nlp_resolved") or "").strip(),
                "nlp_source":        (r.get("nlp_source") or "").strip(),
                "nlp_latency_ms":    _to_float(r.get("nlp_latency_ms")),
                "nlp_correct":       _to_bool(r.get("nlp_correct")),
                "e2e_verified":      _to_bool(r.get("e2e_verified")),
                "e2e_elapsed_sec":   _to_float(r.get("e2e_elapsed_sec")),
                "dispatched_count":  _to_int(r.get("dispatched_count")),
            })
    return rows


def load_runs(csv_paths: List[Path]) -> List[List[Dict]]:
    runs = []
    for p in csv_paths:
        rows = load_csv(p)
        runs.append(rows)
        print(f"  loaded {p.name}: {len(rows)} commands")
    return runs


# ---------------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------------

def per_run_headline(rows: List[Dict]) -> Dict:
    n = len(rows)
    nlp_correct  = sum(1 for r in rows if r["nlp_correct"])
    dispatched   = sum(1 for r in rows if r["dispatched_count"] > 0)
    write_ok     = sum(1 for r in rows if r["e2e_verified"])
    e2e_strict   = sum(1 for r in rows if r["nlp_correct"] and r["e2e_verified"])
    wrong_target = sum(1 for r in rows if r["e2e_verified"] and not r["nlp_correct"])

    return {
        "n":                  n,
        "nlp_acc":            nlp_correct / n if n else 0,
        "dispatch":           dispatched / n if n else 0,
        "write_ok_full":      write_ok / n if n else 0,
        "write_ok_dispatched": write_ok / dispatched if dispatched else 0,
        "e2e_strict":         e2e_strict / n if n else 0,
        "wrong_target":       wrong_target / n if n else 0,
    }


def fmt_pct(values: List[float], decimals: int = 1) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]*100:.{decimals}f}%"
    m = statistics.mean(values)
    s = statistics.stdev(values)
    return f"{m*100:.{decimals}f}% +/- {s*100:.{decimals}f}"


# ---------------------------------------------------------------------------
# Per-category aggregation
# ---------------------------------------------------------------------------

def per_category_table(runs: List[List[Dict]]) -> Dict[str, Dict]:
    """For each category, collect per-run counts, then mean +/- std."""
    cats = sorted({r["gold"] for r in runs[0]})
    out: Dict[str, Dict] = {}
    for cat in cats:
        per_run = []
        for run in runs:
            sub = [r for r in run if r["gold"] == cat]
            n   = len(sub)
            nc  = sum(1 for r in sub if r["nlp_correct"])
            nd  = sum(1 for r in sub if r["dispatched_count"] > 0)
            nv  = sum(1 for r in sub if r["nlp_correct"] and r["e2e_verified"])
            lat = [r["e2e_elapsed_sec"] for r in sub
                   if r["e2e_verified"] and r["e2e_elapsed_sec"] is not None]
            per_run.append({
                "n": n, "nc": nc, "nd": nd, "nv": nv,
                "median_lat": statistics.median(lat) if lat else float("nan"),
            })

        def stat(key):
            vals = [d[key] for d in per_run]
            return (statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0)
        m_nc, s_nc = stat("nc")
        m_nd, s_nd = stat("nd")
        m_nv, s_nv = stat("nv")
        m_lat, s_lat = stat("median_lat")
        out[cat] = {
            "n":       per_run[0]["n"],
            "nc_mean": m_nc, "nc_std": s_nc,
            "nd_mean": m_nd, "nd_std": s_nd,
            "nv_mean": m_nv, "nv_std": s_nv,
            "lat_mean": m_lat, "lat_std": s_lat,
        }
    return out


# ---------------------------------------------------------------------------
# Per-variant subset
# ---------------------------------------------------------------------------

def per_variant_subset(runs: List[List[Dict]]) -> Dict[str, Dict]:
    """Easy (a-d) vs adversarial (e-g) split, with mean +/- std across runs."""
    easy_v = {"a_direct_tr", "b_direct_en", "c_synonym", "d_no_accent"}
    adv_v  = {"e_oov_tr", "f_oov_en", "g_indirect"}

    def subset_metrics(predicate):
        per_run_nlp = []
        per_run_e2e = []
        n_each = None
        for run in runs:
            sub = [r for r in run if predicate(r)]
            if n_each is None:
                n_each = len(sub)
            nc = sum(1 for r in sub if r["nlp_correct"])
            nv = sum(1 for r in sub if r["nlp_correct"] and r["e2e_verified"])
            per_run_nlp.append(nc / len(sub) if sub else 0)
            per_run_e2e.append(nv / len(sub) if sub else 0)
        return n_each, per_run_nlp, per_run_e2e

    n_easy, nlp_easy, e2e_easy = subset_metrics(lambda r: r["variant"] in easy_v)
    n_adv,  nlp_adv,  e2e_adv  = subset_metrics(lambda r: r["variant"] in adv_v)

    return {
        "easy_n":       n_easy,
        "easy_nlp":     nlp_easy,
        "easy_e2e":     e2e_easy,
        "adversarial_n":   n_adv,
        "adversarial_nlp": nlp_adv,
        "adversarial_e2e": e2e_adv,
    }


# ---------------------------------------------------------------------------
# Flakiness: commands that flip across runs
# ---------------------------------------------------------------------------

def flakiness_report(runs: List[List[Dict]]) -> List[Dict]:
    """A command is 'flaky' if nlp_correct or e2e_verified is not unanimous
    across runs."""
    if len(runs) < 2:
        return []
    by_input: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        for r in run:
            by_input[r["input"]].append(r)

    flaky = []
    for inp, occurrences in by_input.items():
        nlp_vals = {r["nlp_correct"] for r in occurrences}
        e2e_vals = {bool(r["e2e_verified"]) for r in occurrences}
        resolved = {r["nlp_resolved"] or "" for r in occurrences}
        if len(nlp_vals) > 1 or len(e2e_vals) > 1 or len(resolved) > 1:
            flaky.append({
                "input":     inp,
                "gold":      occurrences[0]["gold"],
                "variant":   occurrences[0]["variant"],
                "n_runs":    len(occurrences),
                "resolved":  sorted(resolved),
                "nlp_correct_runs": [r["nlp_correct"] for r in occurrences],
                "e2e_verified_runs": [bool(r["e2e_verified"]) for r in occurrences],
                "tiers":     [r["nlp_source"] for r in occurrences],
            })
    return flaky


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_per_category_csv(table: Dict[str, Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "n",
                    "nlp_correct_mean", "nlp_correct_std",
                    "dispatched_mean",   "dispatched_std",
                    "e2e_strict_mean",   "e2e_strict_std",
                    "median_latency_mean", "median_latency_std"])
        for cat, d in table.items():
            w.writerow([cat, d["n"],
                        f"{d['nc_mean']:.2f}", f"{d['nc_std']:.2f}",
                        f"{d['nd_mean']:.2f}", f"{d['nd_std']:.2f}",
                        f"{d['nv_mean']:.2f}", f"{d['nv_std']:.2f}",
                        f"{d['lat_mean']:.3f}", f"{d['lat_std']:.3f}"])



# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_csv_paths(args) -> List[Path]:
    paths: List[Path] = []
    if args.manifest:
        m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        paths = [Path(r["csv_path"]) for r in m.get("runs", [])
                 if r.get("returncode") == 0]
    elif args.csv_dir:
        paths = sorted(Path(args.csv_dir).glob("*.csv"))
    elif args.csvs:
        paths = [Path(p) for p in args.csvs]
    else:
        sys.exit("Provide one of: --manifest, --csv-dir, or --csvs ...")

    paths = [p for p in paths if p.exists()]
    if not paths:
        sys.exit("No CSV files found.")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="Path to manifest.json from run_multi.py")
    ap.add_argument("--csv-dir",  help="Directory containing per-run CSVs")
    ap.add_argument("--csvs",     nargs="+", help="Explicit list of CSV files")
    ap.add_argument("--out-csv",  default="aggregated_per_category.csv")
    ap.add_argument("--flaky-csv", default="flaky_commands.csv")
    args = ap.parse_args()

    paths = resolve_csv_paths(args)
    print(f"Aggregating {len(paths)} run(s):")
    runs = load_runs(paths)

    # --- per-run headline ---
    headlines = [per_run_headline(rows) for rows in runs]
    # collect parallel arrays for mean/std
    by_metric: Dict[str, List[float]] = defaultdict(list)
    for h in headlines:
        for k, v in h.items():
            if isinstance(v, (int, float)) and k != "n":
                by_metric[k].append(v)

    print("\n" + "=" * 72)
    print(f"HEADLINE METRICS (mean +/- std across {len(runs)} run(s))")
    print("=" * 72)
    print(f"  N (per run)                : {headlines[0]['n']}")
    print(f"  NLP accuracy               : {fmt_pct(by_metric['nlp_acc'])}")
    print(f"  Dispatch rate              : {fmt_pct(by_metric['dispatch'])}")
    print(f"  Write success (full)       : {fmt_pct(by_metric['write_ok_full'])}")
    print(f"  Write success (dispatched) : {fmt_pct(by_metric['write_ok_dispatched'])}")
    print(f"  Strict E2E accuracy        : {fmt_pct(by_metric['e2e_strict'])}")
    print(f"  Writes to wrong target     : {fmt_pct(by_metric['wrong_target'])}")

    # --- per-tier (rules / ollama / unresolved) ---
    print("\n" + "-" * 72)
    print("PER-TIER (mean +/- std across runs)")
    print("-" * 72)
    for tier in ["rules", "ollama", "unresolved"]:
        per_run_count = []
        per_run_acc   = []
        per_run_lat   = []
        for run in runs:
            sub = [r for r in run if r["nlp_source"] == tier]
            per_run_count.append(len(sub))
            per_run_acc.append(sum(1 for r in sub if r["nlp_correct"]) / len(sub) if sub else 0)
            lats = [r["nlp_latency_ms"] for r in sub if r["nlp_latency_ms"] is not None]
            per_run_lat.append(statistics.median(lats) if lats else 0)
        n_mean = statistics.mean(per_run_count)
        n_std  = statistics.stdev(per_run_count) if len(per_run_count) > 1 else 0
        print(f"  {tier:10s} n={n_mean:.1f}+/-{n_std:.1f}    "
              f"acc={fmt_pct(per_run_acc)}    "
              f"median_lat={statistics.mean(per_run_lat):.2f}+/-"
              f"{statistics.stdev(per_run_lat) if len(per_run_lat)>1 else 0:.2f} ms")

    # --- variant subsets (easy a-d vs adversarial e-g) ---
    print("\n" + "-" * 72)
    print("VARIANT SUBSETS (easy subset versus out-of-vocabulary/indirect subset)")
    print("-" * 72)
    sub = per_variant_subset(runs)
    print(f"  Easy subset (variants a-d, n={sub['easy_n']}):")
    print(f"    NLP accuracy   = {fmt_pct(sub['easy_nlp'])}")
    print(f"    Strict E2E acc = {fmt_pct(sub['easy_e2e'])}")
    print(f"  Adversarial subset (variants e-g, n={sub['adversarial_n']}):")
    print(f"    NLP accuracy   = {fmt_pct(sub['adversarial_nlp'])}")
    print(f"    Strict E2E acc = {fmt_pct(sub['adversarial_e2e'])}")

    # --- per-category ---
    table = per_category_table(runs)
    print("\n" + "-" * 72)
    print("PER-CATEGORY (mean +/- std across runs)")
    print("-" * 72)
    print(f"  {'cat':10s} {'N':>3s} {'NLPc':>10s} {'Disp':>10s} {'E2Estr':>10s} {'medLat':>12s}")
    for cat, d in table.items():
        print(f"  {cat:10s} {d['n']:>3d} "
              f"{d['nc_mean']:>4.1f}+/-{d['nc_std']:<3.1f} "
              f"{d['nd_mean']:>4.1f}+/-{d['nd_std']:<3.1f} "
              f"{d['nv_mean']:>4.1f}+/-{d['nv_std']:<3.1f} "
              f"{d['lat_mean']:>5.2f}+/-{d['lat_std']:<4.2f}s")

    # --- flakiness ---
    flaky = flakiness_report(runs)
    print("\n" + "-" * 72)
    print(f"FLAKY COMMANDS  (changed across runs): {len(flaky)}")
    print("-" * 72)
    for f in flaky[:20]:  # cap output, full list goes to CSV
        print(f"  [{f['gold']:9s}/{f['variant']:14s}] resolved={f['resolved']}  "
              f"NLP={f['nlp_correct_runs']}  E2E={f['e2e_verified_runs']}")
        print(f"     input: {f['input']}")
    if len(flaky) > 20:
        print(f"  ... and {len(flaky) - 20} more (full list in {args.flaky_csv})")

    # --- write output files ---
    write_per_category_csv(table, Path(args.out_csv))
    if flaky:
        with Path(args.flaky_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gold", "variant", "input",
                        "resolved_set",
                        "nlp_correct_per_run", "e2e_verified_per_run",
                        "tiers_per_run"])
            for fl in flaky:
                w.writerow([fl["gold"], fl["variant"], fl["input"],
                            "|".join(fl["resolved"]),
                            ",".join(str(b) for b in fl["nlp_correct_runs"]),
                            ",".join(str(b) for b in fl["e2e_verified_runs"]),
                            ",".join(fl["tiers"])])

    print("\n" + "=" * 72)
    print(f"Outputs written:")
    print(f"  {args.out_csv}")
    if flaky:
        print(f"  {args.flaky_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
