#!/usr/bin/env python3
"""Coverage vs 643 for EVERY cell of the big table (all methods x 3 budgets).
Per run: unique judged tasks (dedup; corrupt sessions dropped). Prints
missing = 643 - judged per run, so we know exactly which cells are short."""
import os, glob, json
from concurrent.futures import ThreadPoolExecutor
E="/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e"
DP=f"{E}/results/mixmax_wn_sweep/sglang_packed_distinctpage"
OLD=f"{E}/results/mixmax_wn_sweep/sglang_packed"
SS=f"{E}/results/sweep_sparsity"
def rv(jp):
    try: v=json.load(open(jp)).get("verdict")
    except Exception: return None
    return v if v in (True,False) else None   # judge failure (None) = unjudged, never a fail
def sess(runroot):
    best={}
    for s in sorted(glob.glob(f"{runroot}/*/")):
        jps=glob.glob(f"{s}*/judge.json")
        if not jps: continue
        with ThreadPoolExecutor(max_workers=32) as ex:
            rows=list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp)),os.path.getmtime(jp),rv(jp)),jps))
        jd=[v for _,_,v in rows if v is not None]
        if len(jd)>=50 and 100*sum(jd)/len(jd)<5: continue
        for t,m,v in rows:
            if v is None: continue
            if t not in best or m>best[t][0]: best[t]=(m,v)
    return len(best)
def flat(d):
    jps=glob.glob(f"{d}/webvoyager__*/judge.json")
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows=list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp)),rv(jp)),jps))
    return len({t for t,v in rows if v is not None})
def row(label, per_run):  # per_run: list of (vtx, rest)
    cells=[f"{v+r}({643-v-r:+d})" if (v+r)!=643 else "643" for v,r in per_run]
    miss=sum(643-v-r for v,r in per_run)
    print(f"  {label:36} " + "  ".join(f"{c:>10}" for c in cells) + f"   missing={miss}")
print(f"  {'cell':36} {'run1':>10}  {'run2':>10}  {'run3':>10}")
print("=== BSA distinct-page ===")
for S,ps in (("A","64"),("B","32"),("C","32")):
    for rt,rl in (("r0p0625","6.25%"),("r0p125","12.5%")):
        pr=[(sess(f"{DP}/set{S}_{rt}_ps{ps}_vtx/run{r}"),sess(f"{DP}/set{S}_{rt}_ps{ps}_rest/run{r}")) for r in (1,2,3)]
        row(f"BSA-{S} distinct {rl}", pr)
print("=== BSA old chunk-nominal ===")
for S,ps in (("A","64"),("B","32")):
    for rt,rl in (("r0p03125","3.125%"),("r0p0625","6.25%")):
        pr=[(sess(f"{OLD}/set{S}_{rt}_ps{ps}/run{r}"),sess(f"{OLD}/set{S}_{rt}_ps{ps}_rest/run{r}")) for r in (1,2,3)]
        row(f"BSA-{S} old {rl}", pr)
print("=== baselines ===")
def base(mode,bos,b,label):
    for rt,rl in (("r31","3.125%"),("r62","6.25%"),("r125","12.5%")):
        vd=[x for x in sorted(glob.glob(f"{SS}/{mode}_{rt}_{b}_{bos}_*")) if "_rest_" not in x]
        rd=sorted(glob.glob(f"{SS}/{mode}_{rt}_{b}_{bos}_rest_*"))
        vs=[flat(d) for d in vd]; rs=[flat(d) for d in rd]
        vs=[n for n in vs if n>=170][:3]; rs=[n for n in rs if n>=376][:3]
        k=min(len(vs),len(rs))
        if k==0: print(f"  {label+' '+rl:36}   --no healthy runs--"); continue
        row(f"{label} {rl}", [(vs[i],rs[i]) for i in range(k)])
base("quest","bos0eos1","b32","quest b32"); base("quest","bos0eos1","b64","quest b64")
base("gqa_block_sparse","bos1eos1","b32","gqa_block_sparse b32"); base("gqa_block_sparse","bos1eos1","b64","gqa_block_sparse b64")
base("block_sparse","bos1eos1","b32","block_sparse(meanpool) b32"); base("block_sparse","bos1eos1","b64","block_sparse(meanpool) b64")
print("=== dense ===")
for d in sorted(glob.glob(f"{SS}/dense_full*")):
    if "sub200" in d: continue
    n=flat(d); print(f"  {os.path.basename(d):36} {n}({643-n:+d})")
