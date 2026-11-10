#!/usr/bin/env python3
"""Tabulate the distinct-page packed sweep (A/B/C x 6.25%/12.5%), full-643.
Dedup by task across healthy sessions; exclude serving-corrupted sessions
(>=50 judged & <5% pass). Pair vtx-200 + rest-443 per run for full-643; avg of
per-run rates + best-of-3 (task passes if any run passed)."""
import os, json, glob
from concurrent.futures import ThreadPoolExecutor

PK = "results/mixmax_wn_sweep/sglang_packed_distinctpage"
SET = {"A":"ps64","B":"ps32","C":"ps32"}
RATIO = {"6.25%":"r0p0625","12.5%":"r0p125"}

def rv(jp):
    try: v=json.load(open(jp)).get("verdict")
    except Exception: return None
    return v if v in (True,False) else None   # judge failure (None) = unjudged, never a fail

def run_verdicts(cell, run):
    best={}
    for s in sorted(glob.glob(f"{PK}/{cell}/run{run}/*/")):
        jps=glob.glob(f"{s}*/judge.json")
        if not jps: continue
        with ThreadPoolExecutor(max_workers=32) as ex:
            rows=list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp)),os.path.getmtime(jp),rv(jp)),jps))
        jd=[v for _,_,v in rows if v is not None]
        if len(jd)>=50 and 100*sum(jd)/len(jd)<5: continue  # corrupt
        for t,m,v in rows:
            if v is None: continue
            if t not in best or m>best[t][0]: best[t]=(m,v)
    return {t:v for t,(m,v) in best.items()}

print(f"{'setting':22} {'ratio':7} | {'run1':>11} {'run2':>11} {'run3':>11} | {'avg':>6} {'bo3':>6}")
print("-"*88)
for S in ("A","B","C"):
    ps=SET[S]
    for rlabel,rtok in RATIO.items():
        runs=[]
        for run in (1,2,3):
            v=run_verdicts(f"set{S}_{rtok}_{ps}_vtx",run)
            r=run_verdicts(f"set{S}_{rtok}_{ps}_rest",run)
            m={f"v:{k}":val for k,val in v.items()}; m.update({f"r:{k}":val for k,val in r.items()})
            runs.append(m)
        rates=[100*sum(m.values())/len(m) for m in runs if m]
        cells=[f"{(100*sum(m.values())/len(m)) if m else 0:4.1f}%({len(m)})" for m in runs]
        allt=set().union(*[set(m) for m in runs]) if runs else set()
        bo3=100*sum(1 for t in allt if any(m.get(t) for m in runs))/len(allt) if allt else 0
        avg=sum(rates)/len(rates) if rates else 0
        print(f"set{S} ({ps}/{'max32' if S!='B' else 'max16'})".ljust(22)+f" {rlabel:7} | {cells[0]:>11} {cells[1]:>11} {cells[2]:>11} | {avg:5.1f}% {bo3:5.1f}%")
print("\n(n)=unique full-643 tasks judged (target 643). avg=mean per-run rate. bo3=task passes if any run.")
