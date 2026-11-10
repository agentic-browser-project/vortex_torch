#!/bin/bash
#SBATCH --job-name=rest-geom
#SBATCH --output=/vast/projects/liuv/pennnetworks/jiaheng/bsa_val/sweep_%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# Geometry sweep for quest / block_sparse on vortex_torch + sglang.
#   page_size = 64 (fixed); block_size in {32, 64}; ratio in {0.03125, 0.0625, 0.125}
# KV-block reservations are arguments: topk_val / bos / eos.
#   eos MUST be >= 1 (flashinfer applies kv_last_page_len to the final slot).
# Records, in <out>/timing.json:
#   - ttft_s / tpot_s at batch 1 for 3 prompt lengths (unique nonce per request so
#     the radix prefix cache cannot short-circuit prefill and fake a low TTFT)
#   - end-to-end wall clock of the agent phase
# Plus <out>/metrics.txt: sglang Prometheus counters over the real agent workload.
#
# Usage: sbatch --job-name=<n> run_sweep_geom.sbatch <quest|block_sparse> <ratio> <block_size> [bos] [eos] [topk_val]
set -uo pipefail
MODE="$1"; TOPK_RATIO="$2"; BLOCK_SIZE="${3:-32}"
BOS="${4:-1}"; EOS="${5:-1}"; TOPK_VAL="${6:-0}"
PAGE_SIZE=64
PCT=$(python3 -c "print(f'{float(\"$TOPK_RATIO\")*1000:.0f}')")

REPO=/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention
VORTEX_DIR=/vast/projects/liuv/pennnetworks/jiaheng/vortex_torch
SIM_DIR=/vast/projects/liuv/pennnetworks/jiaheng/browser-simulator/browser-use
VPY=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/vortex-bench/bin/python
WVPY=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/webvoyager/bin/python
MODEL=/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-32B
APPT=/vast/parcc/spack/sw/apps/linux-sapphirerapids/apptainer-1.4.1-qpr4lterya7ontg7pkqrx7b3jkab3lcw/bin/apptainer
SIF=/vast/projects/liuv/pennnetworks/jiaheng/bsa_val/xvfb.sif
SUBSET=/vast/projects/liuv/pennnetworks/jiaheng/bsa_val/subset_rest443.json
SLACK=/vast/projects/liuv/pennnetworks/slack_notify.sh
BATCH=32
OUT=${REPO}/results/webvoyager_e2e/results/sweep_sparsity/${MODE}_r${PCT}_b${BLOCK_SIZE}_bos${BOS}eos${EOS}_rest_${SLURM_JOB_ID}
SRVLOG=$OUT/server.log
mkdir -p "$OUT"

notify(){ [ -x "$SLACK" ] && "$SLACK" "$1" >/dev/null 2>&1 || true; }
finish(){
  local rc=$?
  local n; n=$(find "$OUT" -maxdepth 1 -name 'webvoyager__*' -type d 2>/dev/null | wc -l)
  local pr; pr=$(python3 -c "
import glob,json
ok=tot=0
for f in glob.glob('$OUT/webvoyager__*/judge.json'):
    try: v=json.load(open(f)).get('verdict')
    except: continue
    if v is None: continue
    tot+=1; ok+=1 if v else 0
print(f'{100*ok/tot:.1f}% ({ok}/{tot})' if tot else 'ungraded')" 2>/dev/null || echo ungraded)
  local tm; tm=$(python3 -c "
import json
try:
    d=json.load(open('$OUT/timing.json'))
    print(f\"ttft={d['summary']['ttft_s_at_25k']:.2f}s tpot={d['summary']['tpot_ms_at_25k']:.1f}ms e2e={d['summary']['e2e_minutes']:.0f}min\")
except Exception: print('timing n/a')" 2>/dev/null)
  if [ "$rc" -eq 0 ] && [ "$n" -ge 420 ]; then
    notify "✅ geom-REST DONE $MODE r=$TOPK_RATIO blk=$BLOCK_SIZE bos=$BOS eos=$EOS | tasks=$n/200 pass=$pr | $tm | job=$SLURM_JOB_ID"
  else
    notify "❌ geom-REST ENDED rc=$rc $MODE r=$TOPK_RATIO blk=$BLOCK_SIZE bos=$BOS eos=$EOS | tasks=$n/200 pass=$pr | $tm | job=$SLURM_JOB_ID"
  fi
}
trap finish EXIT

echo "HOST: $(hostname)  MODE=$MODE ratio=$TOPK_RATIO page=$PAGE_SIZE block=$BLOCK_SIZE bos=$BOS eos=$EOS topk_val=$TOPK_VAL"
echo "OUT: $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
  quest)        MN=gqa_quest_sparse_attention_sub; MP=submissions/_flow_algorithms_test/gqa_quest_sparse_attention.py ;;
  block_sparse) MN=block_sparse_attention_sub;     MP=submissions/_flow_algorithms_test/block_sparse_attention.py ;;
  gqa_block_sparse) MN=gqa_block_sparse_attention_sub; MP=submissions/_flow_algorithms_test/gqa_block_sparse_attention.py ;;
  *) echo "FATAL: mode must be quest|block_sparse|gqa_block_sparse"; exit 2 ;;
esac
CFG=$(mktemp /tmp/geom_${MODE}_XXXX.json)
cat > "$CFG" <<JSON
{"impl_backend":"triton","attention_backend":"flashinfer","block_size":${BLOCK_SIZE},
 "topk_val":${TOPK_VAL},"topk_ratio":${TOPK_RATIO},
 "block_reserved_bos":${BOS},"block_reserved_eos":${EOS},
 "workload_chunk_size":32,"layers_skip":[],
 "module_name":"${MN}","module_path":"${MP}",
 "dtype":"bfloat16","compilation_cache_dir":"~/.vortex_compilation_cache"}
JSON
echo "[cfg] $(cat $CFG)"; cp "$CFG" "$OUT/vortex_config.json"

DISP=$(( 100 + (SLURM_JOB_ID % 90) )); rm -f /tmp/.X${DISP}-lock 2>/dev/null || true
"$APPT" exec "$SIF" Xvfb :$DISP -screen 0 1280x1024x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
XVFB_PID=$!; sleep 8
[ -S /tmp/.X11-unix/X$DISP ] || { echo "FATAL: Xvfb failed"; cat "$OUT/xvfb.log"; exit 3; }
export DISPLAY=:$DISP

PORT=$(( 21000 + (SLURM_JOB_ID % 20000) ))
PORT=$($VPY -c "
import socket
for p in range($PORT,$PORT+300):
    s=socket.socket()
    try: s.bind(('127.0.0.1',p)); s.close(); print(p); break
    except OSError: continue")
export LD_LIBRARY_PATH=/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/env312/lib:${LD_LIBRARY_PATH:-}
export SGLANG_DISABLE_CUDNN_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$VORTEX_DIR"
$VPY -c '
import os,sys,vortex_torch
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
sa=prepare_server_args(sys.argv[1:])
try: run_server(sa)
finally: kill_process_tree(os.getpid(), include_parent=False)
' --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --attention-backend flashinfer \
  --vortex-config "$(cat "$CFG")" --page-size ${PAGE_SIZE} --context-length 32768 \
  --mem-fraction-static 0.9 --served-model-name "$MODE" --enable-metrics \
  > "$SRVLOG" 2>&1 &
SRV=$!
for i in $(seq 1 240); do
  curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { echo "[serve] ready after $((i*5))s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -60 "$SRVLOG"; kill $XVFB_PID; exit 4; }
  sleep 5
done

# ---- latency probe: TTFT / TPOT at batch 1, before the agent load ----
echo "[timing] latency probe..."
$VPY - "$PORT" "$OUT" <<'PYEOF'
import json,sys,time,urllib.request,uuid
port,out=sys.argv[1],sys.argv[2]
URL=f"http://127.0.0.1:{port}/v1/chat/completions"
def call(prompt,mt):
    body=json.dumps({"model":"probe","messages":[{"role":"user","content":prompt}],
                     "max_tokens":mt,"temperature":0}).encode()
    t0=time.perf_counter()
    r=urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"}),timeout=900)
    d=json.loads(r.read()); t=time.perf_counter()-t0
    return t,d.get("usage",{})
# ~4 chars/token of filler; unique nonce per request defeats the radix prefix cache,
# which would otherwise skip prefill and report an artificially low TTFT.
unit="The maintenance log records routine inspection of the facility equipment. "
res={}
NGEN=65
for label,ntok in (("4k",4000),("16k",16000),("25k",25000)):
    reps=[]
    for rep in range(2):
        base=f"[{uuid.uuid4().hex}] "+unit*int(ntok/18)
        t1,u1=call(base+"\nReply with one word.",1)
        base2=f"[{uuid.uuid4().hex}] "+unit*int(ntok/18)
        t2,u2=call(base2+"\nWrite a short paragraph.",NGEN)
        gen=u2.get("completion_tokens",NGEN)
        tpot=(t2-t1)/max(1,gen-1)
        reps.append({"ttft_s":t1,"total_s":t2,"gen_tokens":gen,"tpot_s":tpot,
                     "prompt_tokens":u1.get("prompt_tokens")})
    res[label]={"reps":reps,
                "ttft_s":sum(r["ttft_s"] for r in reps)/len(reps),
                "tpot_s":sum(r["tpot_s"] for r in reps)/len(reps)}
    print(f"  {label}: TTFT {res[label]['ttft_s']*1000:.0f} ms  TPOT {res[label]['tpot_s']*1000:.2f} ms "
          f"({reps[0]['prompt_tokens']} prompt tok)")
json.dump({"latency_batch1":res},open(f"{out}/timing.json","w"),indent=2)
PYEOF

# ---- agent phase (end-to-end wall clock) ----
cd "$SIM_DIR"
export USE_TSA=1 TSA_BASE_URL="http://127.0.0.1:${PORT}/v1" TSA_MODEL="$MODE" TSA_API_KEY=EMPTY
export SIM_HEADLESS=0 SIM_TASK_IDS="$SUBSET" SIM_MAX_TOKENS=1024
export SIM_JUDGE=1 SIM_JUDGE_MODEL=gemini-3.6-flash SIM_JUDGE_CONCURRENCY=8 SIM_JUDGE_VISION=1
export PLAYWRIGHT_BROWSERS_PATH=/vast/home/j/jiaheng/.cache/ms-playwright
E2E_START=$(date +%s)
echo "[sim] start $(date -Is)"
$WVPY -m simulator capture --no-vision --task-num 700 --batch-size $BATCH \
  --source webvoyager --shuffle --max-steps 15 --task-timeout 5400 --llm-timeout 1200 \
  --out-dir "$OUT" 2>&1 | tee "$OUT/sim.log"
E2E_END=$(date +%s)
E2E=$((E2E_END-E2E_START))
echo "[sim] end $(date -Is)  elapsed ${E2E}s"

curl -s "http://127.0.0.1:${PORT}/metrics" > "$OUT/metrics.txt" 2>/dev/null || true

python3 - "$OUT" "$E2E" "$MODE" "$TOPK_RATIO" "$BLOCK_SIZE" "$BOS" "$EOS" "$TOPK_VAL" "$PAGE_SIZE" <<'PYEOF'
import json,sys,os,re
out,e2e=sys.argv[1],int(sys.argv[2])
d={}
p=f"{out}/timing.json"
if os.path.exists(p): d=json.load(open(p))
d["config"]=dict(mode=sys.argv[3],topk_ratio=float(sys.argv[4]),block_size=int(sys.argv[5]),
                 bos=int(sys.argv[6]),eos=int(sys.argv[7]),topk_val=int(sys.argv[8]),
                 page_size=int(sys.argv[9]))
d["e2e_seconds"]=e2e
lb=d.get("latency_batch1",{})
# Server-side aggregates over the real agent workload (Prometheus histogram sums).
agg={}
mp=f"{out}/metrics.txt"
if os.path.exists(mp):
    txt=open(mp).read()
    def g(name):
        m=re.search(rf"^{name}\s+([0-9.eE+-]+)$",txt,re.M)
        return float(m.group(1)) if m else None
    for k in ("sglang:time_to_first_token_seconds_sum","sglang:time_to_first_token_seconds_count",
              "sglang:inter_token_latency_seconds_sum","sglang:inter_token_latency_seconds_count",
              "sglang:e2e_request_latency_seconds_sum","sglang:e2e_request_latency_seconds_count",
              "sglang:prompt_tokens_total","sglang:generation_tokens_total"):
        v=g(k)
        if v is not None: agg[k.replace("sglang:","")]=v
    def mean(a,b):
        s,c=agg.get(a),agg.get(b)
        return s/c if s and c else None
    agg["mean_ttft_s_workload"]=mean("time_to_first_token_seconds_sum","time_to_first_token_seconds_count")
    agg["mean_itl_s_workload"]=mean("inter_token_latency_seconds_sum","inter_token_latency_seconds_count")
    agg["mean_e2e_req_s_workload"]=mean("e2e_request_latency_seconds_sum","e2e_request_latency_seconds_count")
d["workload_aggregates"]=agg
d["summary"]={
  "ttft_s_at_25k": lb.get("25k",{}).get("ttft_s"),
  "tpot_ms_at_25k": (lb.get("25k",{}).get("tpot_s") or 0)*1000,
  "ttft_s_at_4k": lb.get("4k",{}).get("ttft_s"),
  "tpot_ms_at_4k": (lb.get("4k",{}).get("tpot_s") or 0)*1000,
  "e2e_minutes": e2e/60.0,
  "mean_ttft_s_workload": agg.get("mean_ttft_s_workload"),
  "mean_itl_ms_workload": (agg.get("mean_itl_s_workload") or 0)*1000,
}
json.dump(d,open(p,"w"),indent=2)
print("[timing] summary:",json.dumps(d["summary"],indent=2))
PYEOF

kill $SRV 2>/dev/null; kill $XVFB_PID 2>/dev/null; rm -f "$CFG"
echo "JOB_DONE $MODE r=$TOPK_RATIO block=$BLOCK_SIZE bos=$BOS eos=$EOS OUT=$OUT"
