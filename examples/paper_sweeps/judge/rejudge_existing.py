#!/usr/bin/env python3
"""Re-judge existing trajectories with the simulator's own gemini-3.6-flash
judge (judge_task_dir), writing judge.json — identical methodology to the
inline SIM_JUDGE path, so verdicts match the runs' existing judge.json.
Skips already-judged dirs. Usage: rejudge_existing.py <run_dir_with_task_subdirs>
"""
import sys, os, glob, asyncio
sys.path.insert(0, "/vast/projects/liuv/pennnetworks/jiaheng/browser-simulator/browser-use")
from pathlib import Path
from simulator.tasks import load_webvoyager_tasks
from simulator.core.judge import build_judge_llm, judge_semaphore, judge_task_dir

RUN = Path(sys.argv[1])
by_folder = {t.folder_name: t for t in load_webvoyager_tasks()}
cfg = build_judge_llm()
sema = judge_semaphore()

async def main():
    dirs = sorted(glob.glob(str(RUN / "webvoyager__*")))
    todo = []
    for d in dirs:
        fn = os.path.basename(d)
        jp = Path(d) / "judge.json"
        if jp.exists():
            try:
                import json
                if json.load(open(jp)).get("verdict") is not None:
                    continue
            except Exception:
                pass
        t = by_folder.get(fn)
        if t is not None:
            todo.append((t, Path(d)))
    print(f"{len(dirs)} dirs, {len(todo)} to (re)judge with {cfg.model}", flush=True)
    await asyncio.gather(*[judge_task_dir(cfg, t, d, sema) for t, d in todo])
    # tally
    import json
    v = [bool(json.load(open(p)).get("verdict")) for p in glob.glob(str(RUN / "webvoyager__*/judge.json"))
         if json.load(open(p)).get("verdict") is not None]
    print(f"DONE: {len(v)} judged | pass {sum(v)} ({100*sum(v)/max(len(v),1):.1f}%)", flush=True)

asyncio.run(main())
