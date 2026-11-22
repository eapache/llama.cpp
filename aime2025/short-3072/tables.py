from pathlib import Path
from collections import Counter
import csv
import gzip
import json
import random
import statistics as st

base = Path(__file__).resolve().parent
def csvrows(p):
    return list(csv.DictReader(p.open()))
def percent(n): return f"{100*n:.2f}%"
def fmt(x):
    return f"{x:g}"
def results():
    lines = ["| Context | Correct, fixed Q4 | Correct, lazy | Lazy minus fixed (95% interval) | Mean output tokens, fixed / lazy | Length stops, fixed / lazy |","| ---: | ---: | ---: | ---: | ---: | ---: |"]
    other = ["| Context | Both correct | Lazy only | Fixed only | Both wrong | Neither cache switched | SWA only | Full only | Both switched |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c,p in [(8192,base.parent),(3072,base)]:
        rr = csvrows(p/"paired-results.csv")
        scores = [sum(r[arm+"_correct"]=="True" for r in rr) for arm in ["fixed","lazy"]]
        ids = sorted({r["problem"] for r in rr})
        vals = [st.mean(int(r["lazy_correct"]=="True")-int(r["fixed_correct"]=="True") for r in rr if r["problem"]==i) for i in ids]
        rng=random.Random(2025)
        boots=sorted(st.mean(rng.choices(vals,k=30))*100 for _ in range(50000))
        tokens = [st.mean(int(r[arm+"_tokens"]) for r in rr) for arm in ("fixed","lazy")]
        stops = [sum(r[arm+"_finish"]=="length" for r in rr) for arm in ("fixed","lazy")]
        n=len(rr)
        lines.append(f"| {c:,} | {scores[0]}/{n} ({percent(scores[0]/n)}) | {scores[1]}/{n} ({percent(scores[1]/n)}) | {100*st.mean(vals):+.2f} pp ({boots[1250]:+.2f}, {boots[48749]:+.2f}) | {tokens[0]:,.0f} / {tokens[1]:,.0f} | {stops[0]} / {stops[1]} |")
        def b(r,key): return r[key]=="True"
        outcomes = Counter((b(r,"fixed_correct"),b(r,"lazy_correct")) for r in rr)
        switches = Counter((b(r,"swa_switched"),b(r,"full_switched")) for r in rr)
        counts = [outcomes[(1,1)], outcomes[(0,1)], outcomes[(1,0)], outcomes[(0,0)], switches[(0,0)],switches[(1,0)],switches[(0,1)],switches[(1,1)]]
        other.append("| "+f"{c:,}"+" | "+" | ".join(map(str, counts))+" |")
    return "\n".join(lines+[""]+other)
def memtable():
    lines=["| Context | Cache policy | CUDA KV buffers, bytes | CUDA KV + rotation, bytes | CUDA model, bytes | CUDA startup compute, bytes | Server VRAM at readiness, MiB | Server peak VRAM, MiB (median [range]) |","| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ctx,source in [(8192,base.parent/"long-memory"),(3072,base)]:
        for arm in ("fixed","lazy"):
            ts=[json.load(gzip.open(p))["summary"]["memory"] for p in (source/"trials").glob(f"*-{arm}.json.gz")]
            ready=[t["ready"]/1048576 for t in ts]
            peak=[t["peak"]/1048576 for t in ts]
            name="8192" if ctx==8192 else "I-10"
            sizes=[list(map(int,l.split())) for l in (base/f"replay/{name}-{arm}.bytes").read_text().splitlines()]
            _,model,cache,compute,_=sizes[0]
            kv=sizes[1][4]
            assert all(row[2]==cache and row[4]==kv for row in sizes[1:]) and 0<kv<=cache
            rmin,rmax=min(ready),max(ready)
            r=fmt(rmin) if rmin==rmax else f"{fmt(rmin)}-{fmt(rmax)}"
            label="Fixed Q4" if arm=="fixed" else "Lazy"
            lines.append(f"| {ctx:,} | {label} | {kv:,} | {cache:,} | {model:,} | {compute:,} | {r} | {fmt(st.median(peak))} [{fmt(min(peak))}-{fmt(max(peak))}] |")
    return "\n".join(lines)
def drifttable():
    rr=csvrows(base/"replay/drift.csv")
    lines=["| Full-attention cache phase | Checkpoints | Mean KL, fixed Q4 | Mean KL, lazy | Top-1 matches to Q8, fixed / lazy |","| --- | ---: | ---: | ---: | ---: |"]
    for pre in [True,False]:
        groups=[[r for r in rr if r["arm"]==a and (int(r["prefix_tokens"])<=1536)==pre] for a in ("fixed","lazy")]
        n=len(groups[0])
        k=[st.mean(float(r["kl_from_q8_nats"]) for r in g) for g in groups]
        match=[sum(int(r["top1_match"]) for r in g) for g in groups]
        lines.append(f"| {'Through row 1,536' if pre else 'After full conversion'} | {n} | {k[0]:.6f} | {k[1]:.6f} | {match[0]}/{n} / {match[1]}/{n} |")
    return "\n".join(lines)
if __name__=="__main__":
    print(results(),"\n\n",drifttable(),"\n\n",memtable())
