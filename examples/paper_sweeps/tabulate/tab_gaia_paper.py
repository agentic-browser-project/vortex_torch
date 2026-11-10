#!/usr/bin/env python3
"""GAIA best-of-3 success (90 tasks, page size 32, selection ratios 6.25% / 12.5%)
for Dense, Quest, Block Sparse, and BrowSA, from runs 1-3 of each cell.

Verdict per task dir: baselines prefer the in-run judge (judge.json "verdict"),
BrowSA prefers the standard success-protocol judge (webvoyager_eval.json "success");
each falls back to the other file. None / missing = unjudged (never a fail)."""
import glob, json, os
E = "/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e/results"
GB = f"{E}/gaia_baselines_3run"
DP = f"{E}/mixmax_wn_sweep/sglang_packed"

def verdict(td, prefer):
    order = [("judge.json", "verdict"), ("webvoyager_eval.json", "success")]
    if prefer == "eval": order.reverse()
    for fn, key in order:
        try: v = json.load(open(f"{td}/{fn}")).get(key)
        except Exception: continue
        if v in (True, False): return v
    return None

def collect(root, prefer):  # task dirs sit directly in root or one session level down
    d = {}
    for td in glob.glob(f"{root}/gaia__*") + glob.glob(f"{root}/*/gaia__*"):
        v = verdict(td, prefer)
        if v is not None: d[os.path.basename(td)] = d.get(os.path.basename(td), False) or v
    return d

def runs(cell, prefer):
    out = []
    for r in (1, 2, 3):
        d = sorted(p for p in glob.glob(f"{cell}/run{r}*") if os.path.basename(p) in (f"run{r}",) or
                   os.path.basename(p)[len(f"run{r}"):].lstrip("_").isdigit())
        out.append(collect(d[0], prefer) if d else {})
    return out

cells = {
    "Dense":              (f"{GB}/dense", "judge"),
    "Quest 6.25%":        (f"{GB}/quest_r62", "judge"),
    "Quest 12.5%":        (f"{GB}/quest_r125_b32", "judge"),
    "Block Sparse 6.25%": (f"{GB}/gqa_block_sparse_r62", "judge"),
    "Block Sparse 12.5%": (f"{GB}/gqa_block_sparse_r125_b32", "judge"),
    "BrowSA 6.25%":       (f"{DP}/dp_gaia_setB_r0p0625", "eval"),
    "BrowSA 12.5%":       (f"{DP}/dp_gaia_setB_r0p125", "eval"),
}
for name, (cell, prefer) in cells.items():
    rs = runs(cell, prefer)
    solved = {t for m in rs for t, v in m.items() if v}
    print(f"{name:20} bo3 {100*len(solved)/90:5.1f}   judged/run {[len(m) for m in rs]}")
