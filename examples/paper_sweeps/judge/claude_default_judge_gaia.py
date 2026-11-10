"""Judge GAIA packed-sweep cells with the simulator's DEFAULT success method
(official WebVoyager prompt + reference guidance + last-2 screenshots),
backed by gemini via Google's OpenAI-compatible endpoint (no DashScope key
on this cluster). Writes webvoyager_eval.json into each task dir (resumable)."""
import os, sys, asyncio
from pathlib import Path

os.environ['USE_TSA'] = '0'
os.environ['DASHSCOPE_API_KEY'] = open(
    '/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/gemini-api-key.txt').read().strip()
sys.path.insert(0, '/vast/projects/liuv/pennnetworks/jiaheng/browser-simulator/browser-use')
import simulator.config as C
import simulator.eval.common as common
GURL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
C.DASHSCOPE_BASE_URL = GURL
common.DASHSCOPE_BASE_URL = GURL
from simulator.eval.success import evaluate_success

sess = Path(sys.argv[1])
res = asyncio.run(evaluate_success(sess, model='gemini-3.6-flash', k=2))
n = len(res)
s = sum(1 for r in res if r.get('success') is True)
u = sum(1 for r in res if r.get('success') is None)
print(f"DEFAULTJUDGE_CELL {sess.parent.parent.name}/{sess.parent.name}: SUCCESS {s}/{n} (unknown {u})")
