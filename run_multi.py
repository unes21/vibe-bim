"""
run_benchmark.py — VIBE Extended Benchmark Harness
===================================================
Runs a controlled, reproducible benchmark against a live VIBE Flask server
and produces both a per-command CSV log and a summary table grouped by
category, language, and resolution tier (rules / ollama / unresolved).


Usage
-----
    1. Start the Flask server:   python app.py
    2. (Optional) Start Ollama:  ollama serve  &&  ollama pull llama3.1:8b
    3. Run:                      python run_benchmark.py

Output
------
    vibe_bench_results.csv   — one row per command (timestamp, input, gold,
                               resolved, source, latency_ms, correct)
    vibe_bench_summary.txt   — aggregate metrics per category, language,
                               and resolution tier

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
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------
# Each category has 7 variants covering a deliberate syntactic spectrum:
#   (a) direct TR keyword      — should resolve at tier 1 (rules)
#   (b) direct EN keyword      — should resolve at tier 1 (rules)
#   (c) TR synonym / phrasing  — tier 1 if alias known, else tier 2/3
#   (d) accent-stripped TR     — tests robustness of rule matcher
#   (e) OOV TR sentence        — designed to miss rules; needs LLM tier
#   (f) OOV EN sentence        — designed to miss rules; needs LLM tier
#   (g) indirect / descriptive — probes LLM semantic understanding
#
# "Gold" is the expected category label. (g)-rows without a keyword are
# intentionally included to measure LLM-only resolution rate.
# ---------------------------------------------------------------------------

TEST_CORPUS: List[Tuple[str, str, str, str]] = [
    # (gold_category, language, variant_tag, command_text)

    # --- ROOF ---
    ("roof", "tr", "a_direct_tr",  "çatıya not ekle"),
    ("roof", "en", "b_direct_en",  "add a note to the roof"),
    ("roof", "tr", "c_synonym",    "çatıların açıklamasına yaz"),
    ("roof", "tr", "d_no_accent",  "cati elemanlarina yorum ekle"),
    ("roof", "tr", "e_oov_tr",     "binanın üst kapağına bir açıklama gir"),
    ("roof", "en", "f_oov_en",     "annotate the top covering element of the building"),
    ("roof", "en", "g_indirect",   "mark the element that sheds rainwater"),

    # --- WALL ---
    ("wall", "tr", "a_direct_tr",  "duvara yorum yaz"),
    ("wall", "en", "b_direct_en",  "write a comment on the wall"),
    ("wall", "tr", "c_synonym",    "bölme elemanlarına not ekle"),
    ("wall", "tr", "d_no_accent",  "duvar elemanlarina aciklama ekle"),
    ("wall", "tr", "e_oov_tr",     "dikey taşıyıcı kaplamalara bilgi gir"),
    ("wall", "en", "f_oov_en",     "annotate the vertical partitions"),
    ("wall", "en", "g_indirect",   "label the enclosing vertical surfaces"),

    # --- FLOOR ---
    ("floor", "tr", "a_direct_tr", "döşeme üzerine not ekle"),
    ("floor", "en", "b_direct_en", "add a note to the floor"),
    ("floor", "tr", "c_synonym",   "zemin plakalarına yorum yaz"),
    ("floor", "tr", "d_no_accent", "doseme elemanlarina aciklama ekle"),
    ("floor", "tr", "e_oov_tr",    "yatay taşıyıcı levhalara açıklama gir"),
    ("floor", "en", "f_oov_en",    "annotate the horizontal slab elements"),
    ("floor", "en", "g_indirect",  "tag the surfaces you walk on"),

    # --- DOOR ---
    ("door", "tr", "a_direct_tr",  "kapıya yorum ekle"),
    ("door", "en", "b_direct_en",  "annotate the doors"),
    ("door", "tr", "c_synonym",    "giriş kanatlarına not yaz"),
    ("door", "tr", "d_no_accent",  "kapi elemanlarina aciklama yaz"),
    ("door", "tr", "e_oov_tr",     "açılan geçit elemanlarına bilgi gir"),
    ("door", "en", "f_oov_en",     "annotate the hinged passage elements"),
    ("door", "en", "g_indirect",   "tag the elements people walk through"),

    # --- WINDOW ---
    ("window", "tr", "a_direct_tr","pencereye not yaz"),
    ("window", "en", "b_direct_en","add a comment to the windows"),
    ("window", "tr", "c_synonym",  "cam açıklıklara yorum ekle"),
    ("window", "tr", "d_no_accent","pencere elemanlarina aciklama ekle"),
    ("window", "tr", "e_oov_tr",   "dış cepheye açılan şeffaf elemanlara not gir"),
    ("window", "en", "f_oov_en",   "annotate the transparent facade openings"),
    ("window", "en", "g_indirect", "label elements that let light through the walls"),

    # --- CEILING ---
    ("ceiling", "tr", "a_direct_tr","tavana not ekle"),
    ("ceiling", "en", "b_direct_en","add a note to the ceiling"),
    ("ceiling", "tr", "c_synonym",  "üst yüzeye yorum yaz"),
    ("ceiling", "tr", "d_no_accent","tavan elemanlarina aciklama ekle"),
    ("ceiling", "tr", "e_oov_tr",   "iç mekân üst kaplamalarına bilgi gir"),
    ("ceiling", "en", "f_oov_en",   "annotate the interior overhead surfaces"),
    ("ceiling", "en", "g_indirect", "mark the surface directly above the room"),

    # --- STAIR ---
    ("stair", "tr", "a_direct_tr", "merdivene yorum yaz"),
    ("stair", "en", "b_direct_en", "annotate the stairs"),
    ("stair", "tr", "c_synonym",   "basamak sistemine açıklama ekle"),
    ("stair", "tr", "d_no_accent", "merdiven elemanlarina not yaz"),
    ("stair", "tr", "e_oov_tr",    "katlar arası geçiş yapılarına bilgi gir"),
    ("stair", "en", "f_oov_en",    "annotate the inter-floor circulation elements"),
    ("stair", "en", "g_indirect",  "tag what you climb to change levels"),

    # --- COLUMN ---
    ("column", "tr", "a_direct_tr","kolona not ekle"),
    ("column", "en", "b_direct_en","add a note to the columns"),
    ("column", "tr", "c_synonym",  "düşey yapısal elemanlara yorum yaz"),
    ("column", "tr", "d_no_accent","kolon elemanlarina aciklama ekle"),
    ("column", "tr", "e_oov_tr",   "dikey taşıyıcı profillere bilgi gir"),
    ("column", "en", "f_oov_en",   "annotate the vertical load-bearing members"),
    ("column", "en", "g_indirect", "tag the elements that transfer roof loads down"),

    # --- BEAM ---
    ("beam", "tr", "a_direct_tr", "kirişe not ekle"),
    ("beam", "en", "b_direct_en", "annotate the beams"),
    ("beam", "tr", "c_synonym",   "yatay taşıyıcılara yorum yaz"),
    ("beam", "tr", "d_no_accent", "kiris elemanlarina aciklama ekle"),
    ("beam", "tr", "e_oov_tr",    "yatay yük taşıyan profillere bilgi gir"),
    ("beam", "en", "f_oov_en",    "annotate the horizontal load-carrying members"),
    ("beam", "en", "g_indirect",  "tag the horizontal spans supporting the slabs"),

    # --- RAILING ---
    ("railing", "tr", "a_direct_tr","korkuluğa not yaz"),
    ("railing", "en", "b_direct_en","annotate the railings"),
    ("railing", "tr", "c_synonym",  "tutamak sistemine yorum ekle"),
    ("railing", "tr", "d_no_accent","korkuluk elemanlarina aciklama yaz"),
    ("railing", "tr", "e_oov_tr",   "düşmeye karşı koruma elemanlarına bilgi gir"),
    ("railing", "en", "f_oov_en",   "annotate the fall-protection elements"),
    ("railing", "en", "g_indirect", "tag the barriers people hold when using stairs"),

    # --- ROOM ---
    ("room", "tr", "a_direct_tr", "odalara yorum ekle"),
    ("room", "en", "b_direct_en", "annotate the rooms"),
    ("room", "tr", "c_synonym",   "mekânlara not yaz"),
    ("room", "tr", "d_no_accent", "oda bilgilerine aciklama ekle"),
    ("room", "tr", "e_oov_tr",    "kapalı iç hacimlere bilgi gir"),
    ("room", "en", "f_oov_en",    "annotate the enclosed interior spaces"),
    ("room", "en", "g_indirect",  "tag the areas defined by four walls"),

    # --- FURNITURE ---
    ("furniture", "tr", "a_direct_tr","mobilyalara yorum ekle"),
    ("furniture", "en", "b_direct_en","annotate the furniture"),
    ("furniture", "tr", "c_synonym",  "iç donatıya not yaz"),
    ("furniture", "tr", "d_no_accent","mobilya elemanlarina aciklama ekle"),
    ("furniture", "tr", "e_oov_tr",   "oda içi kullanıcı eşyalarına bilgi gir"),
    ("furniture", "en", "f_oov_en",   "annotate the movable indoor items"),
    ("furniture", "en", "g_indirect", "tag the chairs, tables and storage units"),

    # --- LIGHT ---
    ("light", "tr", "a_direct_tr","aydınlatmaya yorum ekle"),
    ("light", "en", "b_direct_en","annotate the lighting fixtures"),
    ("light", "tr", "c_synonym",  "lamba elemanlarına not yaz"),
    ("light", "tr", "d_no_accent","aydinlatma elemanlarina aciklama ekle"),
    ("light", "tr", "e_oov_tr",   "mekânı ışıklandıran cihazlara bilgi gir"),
    ("light", "en", "f_oov_en",   "annotate the illumination devices"),
    ("light", "en", "g_indirect", "tag the fixtures that emit light at night"),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def call_category(server: str, text: str, timeout: float) -> Dict:
    """POST /api/llm/category and return the parsed response."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{server}/api/llm/category",
            json={"text": text},
            timeout=timeout,
        )
        data = r.json() if r.status_code == 200 else {}
    except Exception as exc:
        data = {"category": None, "source": "http_error", "error": str(exc)[:200]}
    latency_ms = (time.perf_counter() - t0) * 1000.0
    # server also reports its own latency; we prefer that when available
    if "latency_ms" not in data:
        data["latency_ms"] = round(latency_ms, 2)
    return data


def run_benchmark(
    server: str,
    out_csv: str,
    out_summary: str,
    timeout: float,
    sleep_between: float,
) -> None:
    rows: List[Dict] = []
    for gold, lang, variant, text in TEST_CORPUS:
        resp = call_category(server, text, timeout)
        resolved = resp.get("category")
        source   = resp.get("source")
        latency  = resp.get("latency_ms")
        correct  = (resolved == gold)
        rows.append({
            "gold": gold, "lang": lang, "variant": variant, "input": text,
            "resolved": resolved, "source": source,
            "latency_ms": latency, "correct": correct,
        })
        marker = "✓" if correct else "✗"
        print(f"  {marker} [{gold:9s} / {lang} / {variant:12s}] "
              f"→ {str(resolved):10s} via {str(source):12s}  ({latency} ms)")
        if sleep_between > 0:
            time.sleep(sleep_between)

    # --- write per-command CSV ---
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gold", "lang", "variant", "input",
            "resolved", "source", "latency_ms", "correct",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # --- compute summary ---
    total          = len(rows)
    correct_total  = sum(r["correct"] for r in rows)
    by_source      = Counter(r["source"] for r in rows)
    by_lang_total  = Counter(r["lang"] for r in rows)
    by_lang_ok     = Counter(r["lang"] for r in rows if r["correct"])

    per_cat_total  = Counter(r["gold"] for r in rows)
    per_cat_ok     = Counter(r["gold"] for r in rows if r["correct"])
    per_cat_source = defaultdict(Counter)
    for r in rows:
        per_cat_source[r["gold"]][r["source"]] += 1

    # latency percentiles, split by source
    lat_by_source: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if isinstance(r["latency_ms"], (int, float)):
            lat_by_source[r["source"]].append(float(r["latency_ms"]))

    # --- write summary ---
    with open(out_summary, "w", encoding="utf-8") as f:
        def w(s=""):
            print(s)
            f.write(s + "\n")

        w("=" * 72)
        w(f"VIBE Benchmark Summary — {datetime.now().isoformat(timespec='seconds')}")
        w(f"Server: {server}")
        w(f"Total commands: N = {total}")
        w(f"Overall accuracy: {correct_total}/{total} "
          f"= {100 * correct_total / total:.1f}%")
        w("=" * 72)

        w("\n-- Accuracy by language --")
        for lang in sorted(by_lang_total):
            tot = by_lang_total[lang]
            ok  = by_lang_ok[lang]
            w(f"  {lang}: {ok}/{tot} = {100 * ok / tot:.1f}%")

        w("\n-- Resolution tier distribution --")
        for src, cnt in by_source.most_common():
            w(f"  {str(src):16s} {cnt:3d}  ({100 * cnt / total:.1f}%)")

        w("\n-- Per-category accuracy and tier breakdown --")
        w(f"  {'category':12s} {'acc':>10s}   tier counts")
        for cat in sorted(per_cat_total):
            tot = per_cat_total[cat]
            ok  = per_cat_ok[cat]
            tiers = per_cat_source[cat]
            tier_str = ", ".join(
                f"{src}={n}" for src, n in sorted(tiers.items())
            )
            w(f"  {cat:12s} {ok}/{tot} ({100 * ok / tot:.0f}%)   {tier_str}")

        w("\n-- Latency by resolution tier (ms) --")
        w(f"  {'tier':16s} {'n':>4s} {'min':>8s} {'median':>8s} {'mean':>8s} {'max':>8s}")
        for src in sorted(lat_by_source):
            vals = lat_by_source[src]
            if not vals:
                continue
            w(f"  {src:16s} {len(vals):4d} "
              f"{min(vals):8.1f} {statistics.median(vals):8.1f} "
              f"{statistics.mean(vals):8.1f} {max(vals):8.1f}")

        w("\nOutputs:")
        w(f"  per-command CSV : {out_csv}")
        w(f"  summary         : {out_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VIBE extended benchmark harness")
    parser.add_argument("--server",  default="http://127.0.0.1:5000",
                        help="Flask server base URL (default: %(default)s)")
    parser.add_argument("--out-csv", default="vibe_bench_results.csv",
                        help="Per-command CSV output (default: %(default)s)")
    parser.add_argument("--out-summary", default="vibe_bench_summary.txt",
                        help="Summary text output (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=45.0,
                        help="HTTP request timeout in seconds (default: %(default)s)")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="Sleep between requests in seconds (default: %(default)s)")
    args = parser.parse_args()

    # probe health first
    try:
        h = requests.get(f"{args.server}/api/health", timeout=3).json()
        print("Server health:")
        print(json.dumps(h, indent=2, ensure_ascii=False))
        print()
    except Exception as exc:
        print(f"WARNING: could not reach {args.server}/api/health ({exc})")

    print(f"Running {len(TEST_CORPUS)} commands...\n")
    run_benchmark(
        server=args.server,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        timeout=args.timeout,
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()
