import glob,json,os
from concurrent.futures import ThreadPoolExecutor
E="/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/results/webvoyager_e2e"
DP=f"{E}/results/mixmax_wn_sweep/sglang_packed_distinctpage"; SS=f"{E}/results/sweep_sparsity"
def v(jp):
    try: return json.load(open(jp)).get("verdict","MISSINGKEY")
    except Exception: return "CORRUPT"
cells={}
for S,ps in (("A","64"),("B","32"),("C","32")):
    for r in ("r0p0625","r0p125"):
        cells[f"BSA-{S} {r}"]=glob.glob(f"{DP}/set{S}_{r}_ps{ps}_*/run*/*/*/judge.json")
for m,b in (("quest","bos0eos1"),("gqa_block_sparse","bos1eos1")):
    for bl in ("b32","b64"):
        for r in ("r62","r125"):
            cells[f"{m} {bl} {r}"]=glob.glob(f"{SS}/{m}_{r}_{bl}_{b}_*/webvoyager__*/judge.json")
cells["dense"]=glob.glob(f"{SS}/dense_full*/webvoyager__*/judge.json")
tot=0
for k,jps in cells.items():
    with ThreadPoolExecutor(32) as ex: vs=list(ex.map(v,jps))
    bad=[jp for jp,x in zip(jps,vs) if x not in (True,False)]
    tot+=len(bad)
    print(f"  {k:26} files={len(jps):5}  non-True/False={len(bad)}" + (f"  e.g. {os.path.relpath(bad[0],E)}" if bad else ""))
print("TOTAL None/corrupt verdicts:", tot)
