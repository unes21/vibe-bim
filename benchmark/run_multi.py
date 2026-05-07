"""
run_multi.py - Multi-run wrapper around run_e2e_benchmark.py
============================================================
Repeats the full 91-command end-to-end benchmark N times back-to-back,
producing one timestamped CSV per run, so that mean +/- std can be
summarized across repeated runs.

This script does NOT modify app.py or the harness. It launches the
existing run_e2e_benchmark.py as a subprocess once per run, with
fresh per-run output paths. Ollama's temperature is already set to
0 in app.py; residual nondeterminism (hardware-level, threading)
is what we want to characterise across runs.

Usage
-----
    # 5 runs, default settings, 10-second pause between runs
    python run_multi.py

    # 3 runs against a remote server
    python run_multi.py --runs 3 --server http://192.168.1.10:5000

    # Fast smoke test: 2 runs over only the wall+roof subset
    python run_multi.py --runs 2 --subset wall,roof

    # All artefacts under a labelled directory
    python run_multi.py --label take3 --out-dir runs/

Outputs
-------
    runs/<label>_run01_<timestamp>.csv
    runs/<label>_run01_<timestamp>.txt
    runs/<label>_run02_<timestamp>.csv
    ...
    runs/<label>_manifest.json    # paths + per-run NLP/E2E summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HARNESS = "run_e2e_benchmark.py"


def parse_summary(summary_path: Path) -> dict:
    """Pick out the headline NLP / E2E lines from the harness summary."""
    out = {"nlp_accuracy_pct": None, "e2e_verified_pct": None}
    if not summary_path.exists():
        return out
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("NLP accuracy"):
            try:
                out["nlp_accuracy_pct"] = float(s.split("=")[-1].strip().rstrip("%"))
            except Exception:
                pass
        elif s.startswith("E2E verified"):
            try:
                pct = s.split("=")[1].split("%")[0].strip()
                out["e2e_verified_pct"] = float(pct)
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-run wrapper for VIBE benchmark.")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of full-corpus passes (default: 5).")
    ap.add_argument("--label", default="multirun",
                    help="Filename prefix for this batch (default: multirun).")
    ap.add_argument("--out-dir", default="runs",
                    help="Directory for per-run CSVs and summaries.")
    ap.add_argument("--rest-sec", type=float, default=10.0,
                    help="Pause between runs to let Dynamo settle (default: 10s).")
    # Pass-through arguments forwarded to run_e2e_benchmark.py
    ap.add_argument("--server",        default="http://127.0.0.1:5000")
    ap.add_argument("--subset",        default="")
    ap.add_argument("--wait-sec",      type=float, default=20.0)
    ap.add_argument("--poll-sec",      type=float, default=0.5)
    ap.add_argument("--http-timeout",  type=float, default=45.0)
    ap.add_argument("--sleep-between", type=float, default=0.5)
    ap.add_argument("--python",        default=sys.executable,
                    help="Python interpreter to invoke harness with.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(HARNESS).exists():
        sys.exit(f"Cannot find {HARNESS} in current directory. "
                 "Run this script from the repo root where run_e2e_benchmark.py lives.")

    manifest = {
        "label": args.label,
        "runs_requested": args.runs,
        "server": args.server,
        "subset": args.subset,
        "wait_sec": args.wait_sec,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "harness": HARNESS,
        "runs": [],
    }
    manifest_path = out_dir / f"{args.label}_manifest.json"

    print(f"\n{'='*72}")
    print(f"VIBE multi-run benchmark   |   {args.runs} runs   |   label='{args.label}'")
    print(f"Output directory : {out_dir.resolve()}")
    print(f"Manifest         : {manifest_path.resolve()}")
    print(f"{'='*72}\n")

    for i in range(1, args.runs + 1):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_label = f"{args.label}_run{i:02d}_{ts}"
        csv_path  = out_dir / f"{run_label}.csv"
        sum_path  = out_dir / f"{run_label}.txt"

        cmd = [
            args.python, HARNESS,
            "--server",        args.server,
            "--out-csv",       str(csv_path),
            "--out-summary",   str(sum_path),
            "--wait-sec",      str(args.wait_sec),
            "--poll-sec",      str(args.poll_sec),
            "--http-timeout",  str(args.http_timeout),
            "--sleep-between", str(args.sleep_between),
        ]
        if args.subset:
            cmd += ["--subset", args.subset]

        print(f"\n----- RUN {i}/{args.runs}  ({ts})  -----")
        print("CMD:", " ".join(cmd))
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        elapsed = time.time() - t0

        summary = parse_summary(sum_path) if rc == 0 else {}
        run_record = {
            "run_index":          i,
            "timestamp":          ts,
            "csv_path":           str(csv_path),
            "summary_path":       str(sum_path),
            "returncode":         rc,
            "elapsed_sec":        round(elapsed, 1),
            "nlp_accuracy_pct":   summary.get("nlp_accuracy_pct"),
            "e2e_verified_pct":   summary.get("e2e_verified_pct"),
        }
        manifest["runs"].append(run_record)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

        print(f"\nRun {i} finished: rc={rc}, elapsed={elapsed/60:.1f} min, "
              f"NLP={summary.get('nlp_accuracy_pct')}%, "
              f"E2E={summary.get('e2e_verified_pct')}%")

        if rc != 0:
            print(f"[WARN] Harness exited with non-zero code {rc}. "
                  "Continuing to next run; inspect summary for partial results.")

        if i < args.runs:
            print(f"Resting {args.rest_sec:.0f}s before next run...")
            time.sleep(args.rest_sec)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"All {args.runs} runs complete.")
    print(f"Per-run results:")
    for r in manifest["runs"]:
        nlp = r["nlp_accuracy_pct"]
        e2e = r["e2e_verified_pct"]
        nlp_s = f"{nlp:5.1f}%" if nlp is not None else "  n/a "
        e2e_s = f"{e2e:5.1f}%" if e2e is not None else "  n/a "
        print(f"  run {r['run_index']:02d}   NLP={nlp_s}   E2E={e2e_s}   "
              f"({r['elapsed_sec']/60:.1f} min)   {r['csv_path']}")
    print(f"\nNext step: aggregate variance across runs:")
    print(f"  python aggregate_runs.py --manifest {manifest_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
