"""Paired AIME25 eval. Restart the server for every independent cache."""
from mem import sample
import gzip
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import urllib.request

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
MODEL = Path.home() / "Downloads/gpt-oss-20b-MXFP4.gguf"
QUESTIONS = json.loads((OUT.parent / "questions.json").read_text())
RUNS = OUT / "trials"
RUNS.mkdir(exist_ok=True)
SAMPLING = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.01, reasoning_effort="low", max_tokens=3072)

def request(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer local-aime-eval"}), timeout=timeout) as r:
        return json.load(r)

def trial(q, rep, arm, seed):
    name = f"{rep:02}-{q['id']}-{arm}"
    dest = RUNS / (name + ".json.gz")
    if dest.exists():
        return
    log = RUNS / (name + ".log")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    typ = "q4_0" if arm == "fixed" else "q8_0:q4_0"
    args = [str(REPO / "build/bin/llama-server"), "-m", str(MODEL), "-dev", "CUDA0", "-ngl", "all", "-sm", "none", "-fa", "on", "-c", "3072", "-np", "1", "-kvu", "-ctk", typ, "-ctv", typ, "--no-warmup", "--no-cache-prompt", "--cache-ram", "0", "--no-context-shift", "--host", "127.0.0.1", "--port", str(port), "-a", "gpt-oss-20b", "--log-verbosity", "4", "--api-key", "local-aime-eval"]
    with log.open("w") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env={**{k:v for k,v in os.environ.items() if not k.startswith("LLAMA_")}, "LLAMA_ATTN_ROT_DISABLE":"0"})
        try:
            for _ in range(1800):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {log}")
                try:
                    if request(base + "/health", timeout=1).get("status") == "ok":
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("server startup timeout")
            payload = dict(SAMPLING, model="gpt-oss-20b", seed=seed, messages=[dict(role="user", content=q["prompt"])])
            finish_sampling = sample(proc.pid)
            start = time.monotonic()
            res = request(base + "/v1/chat/completions", payload, timeout=1800)
            memory = finish_sampling()
            elapsed = time.monotonic() - start
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    choice = res["choices"][0]
    msg = choice["message"]
    final = msg.get("content") or ""
    boxes = re.findall(r"\\boxed{([^}]+)}", final, re.I)
    nums = re.findall(r"\b(\d+)\b", final)
    ans = (boxes or nums or [""])[-1].strip()
    raw = log.read_text(errors="replace")
    transitions = re.findall(r"converted KV cache through row (\d+) and expanded (\d+) cells to (\d+)",raw)
    capacities = re.findall(r"initial KV capacity (\d+) cells \(([^)]+)\), final capacity (\d+) cells \(([^)]+)\)", raw)
    if arm == "lazy" and not capacities:
        raise RuntimeError(f"missing lazy startup evidence: {log}")
    summary = dict(memory=memory,id=q["id"], rep=rep, seed=seed, arm=arm, gold=q["gold"], answer=ans, correct=ans==q["gold"], finish=choice["finish_reason"], tokens=res["usage"]["completion_tokens"], prompt_tokens=res["usage"]["prompt_tokens"], elapsed=round(elapsed,3), transitions=transitions, capacities=capacities)
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp,"wt") as f:
        json.dump(dict(summary=summary, request=payload,response=res,argv=args,server_log=raw), f)
    tmp.replace(dest)
    log.unlink()
    print(json.dumps(summary),flush=True)

for rep in range(1):
    indices = list(range(len(QUESTIONS)))
    random.Random(2025+rep).shuffle(indices)
    for pos,i in enumerate(indices):
        for arm in (("fixed","lazy") if (rep+pos)%2==0 else ("lazy","fixed")):
            trial(QUESTIONS[i],rep,arm,2026000+rep*30+i)
