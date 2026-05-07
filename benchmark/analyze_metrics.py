"""
Compute corrected VIBE benchmark metrics by separating four distinct metrics:
  - NLP accuracy        (Nc/N)        : resolved category == gold
  - Dispatch rate       (Nd/N)        : NLP returned valid category AND model has elements
  - Write success       (Nw/Nd)       : Dynamo write completed
  - End-to-end accuracy (Nv/N strict) : NLP correct AND write succeeded
"""

import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("vibe_e2e_results.csv")

# Normalize boolean columns (CSV has "True"/"False" strings + empties)
for col in ["nlp_correct", "e2e_verified"]:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].map({"True": True, "False": False}).fillna(False)

df["dispatched_count"] = pd.to_numeric(df["dispatched_count"], errors="coerce").fillna(0).astype(int)
df["nlp_latency_ms"]   = pd.to_numeric(df["nlp_latency_ms"], errors="coerce")
df["e2e_elapsed_sec"]  = pd.to_numeric(df["e2e_elapsed_sec"], errors="coerce")

N = len(df)
print(f"=== CORPUS SIZE ===")
print(f"N = {N}\n")

# ============================================================
# 1. HEADLINE METRICS (full corpus)
# ============================================================
print("=== HEADLINE METRICS (full corpus, N=91) ===")

nlp_correct      = df["nlp_correct"].sum()
dispatched       = (df["dispatched_count"] > 0).sum()
write_succeeded  = df["e2e_verified"].sum()
e2e_strict       = (df["nlp_correct"] & df["e2e_verified"]).sum()

print(f"NLP accuracy        : {nlp_correct}/{N} = {nlp_correct/N*100:.1f}%")
print(f"Dispatch rate       : {dispatched}/{N} = {dispatched/N*100:.1f}%")
print(f"Write 'verified'    : {write_succeeded}/{N} = {write_succeeded/N*100:.1f}%  (legacy headline)")
print(f"E2E accuracy STRICT : {e2e_strict}/{N} = {e2e_strict/N*100:.1f}%  (NEW honest headline)")

# How many "verified" writes were actually to the WRONG category?
wrong_target_writes = (~df["nlp_correct"] & df["e2e_verified"]).sum()
print(f"\n!! Writes confirmed but to WRONG target: {wrong_target_writes}/{N} = {wrong_target_writes/N*100:.1f}%")
print(f"   (these are counted as 'verified' in the legacy 73.6% but are substantively incorrect)")

# Write success conditional on dispatch
disp_subset_verified = df[df["dispatched_count"] > 0]["e2e_verified"].sum()
disp_subset_total    = (df["dispatched_count"] > 0).sum()
print(f"\nWrite success on dispatched subset : {disp_subset_verified}/{disp_subset_total} = {disp_subset_verified/disp_subset_total*100:.1f}%")
print()

# ============================================================
# 2. PER-TIER BREAKDOWN
# ============================================================
print("=== PER-TIER BREAKDOWN ===")
for tier in ["rules", "ollama", "unresolved"]:
    sub = df[df["nlp_source"] == tier]
    n   = len(sub)
    if n == 0:
        continue
    nc  = sub["nlp_correct"].sum()
    e2e = (sub["nlp_correct"] & sub["e2e_verified"]).sum()
    lat = sub["nlp_latency_ms"]
    print(f"  {tier:10s}: n={n:2d}, NLP correct={nc}/{n} ({nc/n*100:.0f}%), "
          f"E2E strict correct={e2e}/{n} ({e2e/n*100:.0f}%), "
          f"latency median={lat.median():.2f}ms")
print()

# ============================================================
# 3. PER-VARIANT BREAKDOWN  (CRITICAL for easy subset framing)
# ============================================================
print("=== PER-VARIANT BREAKDOWN ===")
print("(variants a-d target rules, e-g are adversarial / target LLM)")
print()
variant_order = ["a_direct_tr", "b_direct_en", "c_synonym", "d_no_accent",
                 "e_oov_tr", "f_oov_en", "g_indirect"]
for v in variant_order:
    sub = df[df["variant"] == v]
    n   = len(sub)
    nc  = sub["nlp_correct"].sum()
    e2e = (sub["nlp_correct"] & sub["e2e_verified"]).sum()
    print(f"  {v:14s}: n={n:2d}, NLP={nc}/{n} ({nc/n*100:5.1f}%), "
          f"E2E strict={e2e}/{n} ({e2e/n*100:5.1f}%)")

# Easy subset (a-d) vs adversarial subset (e-g)
easy_mask = df["variant"].isin(["a_direct_tr", "b_direct_en", "c_synonym", "d_no_accent"])
adv_mask  = df["variant"].isin(["e_oov_tr", "f_oov_en", "g_indirect"])

print()
print("=== EASY SUBSET (variants a-d, comparable to prior work) ===")
n_easy   = easy_mask.sum()
nc_easy  = df.loc[easy_mask, "nlp_correct"].sum()
e2e_easy = (df.loc[easy_mask, "nlp_correct"] & df.loc[easy_mask, "e2e_verified"]).sum()
print(f"  N            = {n_easy}")
print(f"  NLP accuracy = {nc_easy}/{n_easy} = {nc_easy/n_easy*100:.1f}%")
print(f"  E2E strict   = {e2e_easy}/{n_easy} = {e2e_easy/n_easy*100:.1f}%")

print()
print("=== ADVERSARIAL SUBSET (variants e-g) ===")
n_adv   = adv_mask.sum()
nc_adv  = df.loc[adv_mask, "nlp_correct"].sum()
e2e_adv = (df.loc[adv_mask, "nlp_correct"] & df.loc[adv_mask, "e2e_verified"]).sum()
print(f"  N            = {n_adv}")
print(f"  NLP accuracy = {nc_adv}/{n_adv} = {nc_adv/n_adv*100:.1f}%")
print(f"  E2E strict   = {e2e_adv}/{n_adv} = {e2e_adv/n_adv*100:.1f}%")
print()

# ============================================================
# 4. PER-CATEGORY TABLE (the corrected Table 1)
# ============================================================
print("=== PER-CATEGORY TABLE (corrected Table 1) ===\n")
print(f"{'Category':12s} {'N':>3s} {'NLPc':>5s} {'Disp':>5s} {'Wver':>5s} {'E2Es':>5s} {'medLat':>7s} {'meanLat':>8s}")
rows = []
for cat in df["gold"].unique():
    sub = df[df["gold"] == cat]
    n   = len(sub)
    nc  = sub["nlp_correct"].sum()
    nd  = (sub["dispatched_count"] > 0).sum()
    nw  = sub["e2e_verified"].sum()
    ne2e = (sub["nlp_correct"] & sub["e2e_verified"]).sum()
    lat = sub.loc[sub["e2e_verified"], "e2e_elapsed_sec"]
    medlat  = lat.median() if len(lat) else float("nan")
    meanlat = lat.mean()   if len(lat) else float("nan")
    rows.append((cat, n, nc, nd, nw, ne2e, medlat, meanlat))
    print(f"{cat:12s} {n:>3d} {nc:>5d} {nd:>5d} {nw:>5d} {ne2e:>5d} {medlat:>7.2f} {meanlat:>8.2f}")

print()
print("Total / weighted means:")
total_n   = sum(r[1] for r in rows)
total_nc  = sum(r[2] for r in rows)
total_nd  = sum(r[3] for r in rows)
total_nw  = sum(r[4] for r in rows)
total_ne2e= sum(r[5] for r in rows)
print(f"{'TOTAL':12s} {total_n:>3d} {total_nc:>5d} {total_nd:>5d} {total_nw:>5d} {total_ne2e:>5d}")
print(f"  NLP corr  : {total_nc}/{total_n} = {total_nc/total_n*100:.1f}%")
print(f"  Dispatch  : {total_nd}/{total_n} = {total_nd/total_n*100:.1f}%")
print(f"  Wver      : {total_nw}/{total_n} = {total_nw/total_n*100:.1f}%   <- legacy 73.6% headline (loose)")
print(f"  E2E strict: {total_ne2e}/{total_n} = {total_ne2e/total_n*100:.1f}%   <- proposed honest headline")

# ============================================================
# 5. LATENCY PERCENTILES (Problem 6)
# ============================================================
print("\n=== LATENCY DISTRIBUTION (Problem 6) ===")
lat_all = df.loc[df["e2e_verified"], "e2e_elapsed_sec"].dropna()
print(f"  n={len(lat_all)} verified writes")
for p in [50, 75, 90, 95, 99]:
    print(f"  p{p:2d} = {np.percentile(lat_all, p):.2f}s")
print(f"  max = {lat_all.max():.2f}s")
print(f"  mean= {lat_all.mean():.2f}s")
print(f"  Excluding outliers > 10s: n={(lat_all<=10).sum()}, mean={lat_all[lat_all<=10].mean():.2f}s")

# Rule-tier latency
rule_lat = df.loc[df["nlp_source"] == "rules", "nlp_latency_ms"]
print(f"\n  Rule-tier NLP latency (n={len(rule_lat)}):")
print(f"    min={rule_lat.min():.4f}ms, median={rule_lat.median():.4f}ms, "
      f"mean={rule_lat.mean():.4f}ms, max={rule_lat.max():.4f}ms")

# Ollama-tier latency
oll_lat = df.loc[df["nlp_source"] == "ollama", "nlp_latency_ms"]
print(f"  Ollama-tier NLP latency (n={len(oll_lat)}):")
print(f"    median={oll_lat.median():.0f}ms, mean={oll_lat.mean():.0f}ms")

# ============================================================
# 6. FAILURE TAXONOMY (Problem 16: Type B explicit table)
# ============================================================
print("\n=== TYPE A: rule-tier substring false positives ===")
type_a = df[(df["nlp_source"] == "rules") & (~df["nlp_correct"])]
print(f"  count = {len(type_a)}")
for _, r in type_a.iterrows():
    print(f"    gold={r['gold']:9s} pred={r['nlp_resolved']:9s} | {r['input']}")

print("\n=== TYPE B: Ollama semantic-neighbour confusion ===")
type_b = df[(df["nlp_source"] == "ollama") & (~df["nlp_correct"]) & (df["nlp_resolved"].notna()) & (df["nlp_resolved"] != "")]
print(f"  count = {len(type_b)}")
for _, r in type_b.iterrows():
    print(f"    gold={r['gold']:9s} pred={str(r['nlp_resolved']):9s} | {r['input']}")

print("\n=== TYPE C: unresolved (Turkish synonym gap) ===")
type_c = df[df["nlp_source"] == "unresolved"]
print(f"  count = {len(type_c)}")
print(f"  by variant: {type_c['variant'].value_counts().to_dict()}")
print(f"  by language: {type_c['lang'].value_counts().to_dict()}")
