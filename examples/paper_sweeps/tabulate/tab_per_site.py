#!/usr/bin/env python3
"""Per-website best-of-3 for the Table-1 cells (12.5%, page/block 32 & 64).
Site = the task-id prefix before '--'. bo3 per site = tasks solved in ANY of
the 3 runs / tasks in that site. The ALL row must reproduce Table 1."""
import os, glob, json, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
E="/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e"
DP=f"{E}/results/mixmax_wn_sweep/sglang_packed_distinctpage"; SS=f"{E}/results/sweep_sparsity"
def rv(jp):
    try: v=json.load(open(jp)).get("verdict")
    except Exception: return None
    return v if v in (True,False) else None   # judge failure (None) = unjudged, never a fail
def sess(runroot):
    best={}
    for s in sorted(glob.glob(f"{runroot}/*/")):
        jps=glob.glob(f"{s}*/judge.json")
        if not jps: continue
        with ThreadPoolExecutor(32) as ex: rows=list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp)),os.path.getmtime(jp),rv(jp)),jps))
        jd=[v for _,_,v in rows if v is not None]
        if len(jd)>=50 and 100*sum(jd)/len(jd)<5: continue
        for t,m,v in rows:
            if v is None: continue
            if t not in best or m>best[t][0]: best[t]=(m,v)
    return {t:v for t,(m,v) in best.items()}
def flat(d):
    jps=glob.glob(f"{d}/webvoyager__*/judge.json")
    with ThreadPoolExecutor(32) as ex: rows=list(ex.map(lambda jp:(os.path.basename(os.path.dirname(jp)),rv(jp)),jps))
    return {t:v for t,v in rows if v is not None}
def bsa_runs(S,ps):
    return [{**sess(f"{DP}/set{S}_r0p125_ps{ps}_vtx/run{r}"),**sess(f"{DP}/set{S}_r0p125_ps{ps}_rest/run{r}")} for r in (1,2,3)]
def base_runs(mode,bos,b):
    vd=[x for x in sorted(glob.glob(f"{SS}/{mode}_r125_{b}_{bos}_*")) if "_rest_" not in x]
    rd=sorted(glob.glob(f"{SS}/{mode}_r125_{b}_{bos}_rest_*"))
    vs=[v for v in (flat(d) for d in vd) if len(v)>=170][:3]; rs=[r for r in (flat(d) for d in rd) if len(r)>=376][:3]
    return [{**vs[i],**rs[i]} for i in range(min(len(vs),len(rs)))]
def dense_runs():
    return [r for r in (flat(d) for d in sorted(glob.glob(f"{SS}/dense_full*")) if "sub200" not in d) if len(r)>=600]
CELLS={"BrowSA p32":bsa_runs("B","32"),"BrowSA p64":bsa_runs("A","64"),
       "Quest b32":base_runs("quest","bos0eos1","b32"),"Quest b64":base_runs("quest","bos0eos1","b64"),
       "BlkSp(GQA) b32":base_runs("gqa_block_sparse","bos1eos1","b32"),"BlkSp(GQA) b64":base_runs("gqa_block_sparse","bos1eos1","b64"),
       "Dense":dense_runs()}
def site(t): return re.sub(r"^webvoyager__","",t).split("--")[0].replace("_"," ")
per=defaultdict(dict); ntask=defaultdict(int)
for name,runs in CELLS.items():
    allt=set().union(*[set(r) for r in runs]); solved=defaultdict(int); cnt=defaultdict(int)
    for t in allt:
        s=site(t); cnt[s]+=1; solved[s]+=any(r.get(t) for r in runs)
    for s in cnt: per[s][name]=100*solved[s]/cnt[s]; ntask[s]=max(ntask[s],cnt[s])
    per["ALL"][name]=100*sum(solved.values())/len(allt); ntask["ALL"]=len(allt)
names=list(CELLS); hdr=f"{'site':22} {'n':>4} | "+" ".join(f"{n:>14}" for n in names)
print(hdr); print("-"*len(hdr))
for s in sorted(k for k in per if k!="ALL")+["ALL"]:
    print(f"{s:22} {ntask[s]:>4} | "+" ".join(f"{per[s].get(n,float('nan')):>13.1f}%" for n in names))
