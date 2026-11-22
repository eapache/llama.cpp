from collections import Counter,defaultdict
from pathlib import Path
import json, gzip, csv, statistics as st

base = Path(__file__).resolve().parent
def rows(folder):
    return [json.load(gzip.open(p))["summary"] for p in sorted((folder/"trials").glob("*.json.gz"))]
def memory(folder):
    measurements = rows(folder)
    for arm in ("fixed","lazy"):
        armrows = [r for r in measurements if r["arm"]==arm]
        if not armrows:
            continue
        for metric in ("ready","peak","end"):
            vals = [r["memory"][metric]/1048576 for r in armrows]
            print(arm,metric,min(vals),st.median(vals),max(vals))
    by = defaultdict(dict)
    for r in measurements:
        by[r["id"]][r["arm"]] = r["memory"]
    pairs = [p for p in by.values() if len(p)==2]
    for metric in ("ready","peak","end"):
        print(metric,"lazy-fixed deltas MiB",Counter((p["lazy"][metric]-p["fixed"][metric])/1048576 for p in pairs))

for folder in (base,base.parent/"long-memory"):
    print(folder.name)
    memory(folder)

for folder in (base,base.parent):
    data = list(csv.DictReader((folder/"paired-results.csv").open()))
    print(folder.name, "full switch paired outcomes")
    for switched in ("True","False"):
        group = [r for r in data if r["full_switched"]==switched]
        print(switched,len(group),Counter((r["fixed_correct"],r["lazy_correct"]) for r in group))
    print("AIME split", [(arm,split,sum(r[arm+"_correct"]=="True" for r in data if r["problem"].split("-")[0]==split)) for arm in ("fixed","lazy") for split in ("I","II")])

replay = list(csv.DictReader((base/"replay/drift.csv").open()))
for stage in ("before","after"):
    for arm in ("fixed","lazy"):
        group = [r for r in replay if (int(r["prefix_tokens"])<=1536)==(stage=="before") and r["arm"]==arm]
        print(stage,arm,len(group),st.mean(float(r["kl_from_q8_nats"]) for r in group),sum(int(r["top1_match"]) for r in group))
        print({t:round(st.mean(float(r["kl_from_q8_nats"]) for r in group if r["trace"]==t),6) for t in sorted(set(r["trace"] for r in group))})
