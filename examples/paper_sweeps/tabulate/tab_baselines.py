#!/usr/bin/env python3
"""Tabulate quest / block_sparse / gqa_block_sparse baselines from
results/sweep_sparsity, full-643 = vtx-200 + rest-443, ratios 6.25% & 12.5%,
block 32 & 64. Run dirs have webvoyager__task/judge.json directly (no sessions).
Some halves have >3 run dirs -> keep only healthy runs (judged>=0.85*target &
pass>=3%), take up to 3. full-643 composite = (200*mean_vtx + 443*mean_rest)/643;
bo3 per half (task passes in any run), composited the same way."""
import os, glob, json
from concurrent.futures import ThreadPoolExecutor

BASE = "results/sweep_sparsity"
MODES = {"quest":"bos0eos1","block_sparse":"bos1eos1","gqa_block_sparse":"bos1eos1"}
RATIOS = {"3.125%":"r31","6.25%":"r62","12.5%":"r125"}

def rv(jp):
    try: v=json.load(open(jp)).get("verdict")
    except Exception: return None
    return v if v in (True,False) else None   # judge failure (None) = unjudged, never a fail

def dir_verdicts(d):
    jps = glob.glob(f"{d}/webvoyager__*/judge.json")
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp))[len("webvoyager__"):], rv(jp)), jps))
    return {t:v for t,v in rows if v is not None}

def half_runs(mode, rtok, b, half, target):
    pat = f"{BASE}/{mode}_{rtok}_{b}_{MODES[mode]}{half}_*"
    runs=[]
    for d in sorted(glob.glob(pat)):
        v = dir_verdicts(d)
        if len(v) >= 0.85*target and (100*sum(v.values())/len(v))>=3:
            runs.append(v)
    return runs[:3]

print(f"{'setting':26} {'budget':7} | {'run1':>7} {'run2':>7} {'run3':>7} | {'avg':>6} {'bo3':>6}")
print("-"*74)
for mode in ("quest","block_sparse","gqa_block_sparse"):
    for b in ("b32","b64"):
        for rl,rt in RATIOS.items():
            vr=half_runs(mode,rt,b,"",200)      # ordered vtx runs
            rr=half_runs(mode,rt,b,"_rest",443) # ordered rest runs
            k=min(len(vr),len(rr))
            if k==0:
                print(f"{mode+' '+b:26} {rl:7} |  --no healthy runs--"); continue
            # pair vtx[i]+rest[i] -> per-run full-643
            perrun=[]; allt=set()
            for i in range(k):
                m={**{f"v:{t}":x for t,x in vr[i].items()}, **{f"r:{t}":x for t,x in rr[i].items()}}
                perrun.append(m); allt|=set(m)
            rates=[100*sum(m.values())/len(m) for m in perrun]
            avg=sum(rates)/len(rates)
            bo3=100*sum(1 for t in allt if any(m.get(t) for m in perrun))/len(allt)
            cells=[f"{x:5.1f}%" for x in rates]+["  -   "]*(3-len(rates))
            print(f"{mode+' '+b:26} {rl:7} | {cells[0]:>7} {cells[1]:>7} {cells[2]:>7} | {avg:5.1f}% {bo3:5.1f}%")
