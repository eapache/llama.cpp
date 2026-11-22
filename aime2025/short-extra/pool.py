"""Pool the two 3,072-token runs, pairing by problem and sampling seed."""
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

out = Path(__file__).resolve().parent
questions = json.loads((out.parent / "questions.json").read_text())
ids = [q["id"] for q in questions]
rows = []
for folder in (out.parent / "short-3072", out):
    with (folder / "paired-results.csv").open() as f:
        rows.extend(csv.DictReader(f))
assert len(rows) == 120
seen = set()
for r in rows:
    rep = int(r["rep"])
    key = (r["problem"], rep)
    assert key not in seen and 0 <= rep < 4
    assert int(r["seed"]) == 2026000 + 30*rep + ids.index(r["problem"])
    seen.add(key)
rows.sort(key=lambda r: (ids.index(r["problem"]), int(r["rep"])))
with (out / "pooled-results.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def stats(group):
    fixed = sum(r["fixed_correct"] == "True" for r in group)
    lazy = sum(r["lazy_correct"] == "True" for r in group)
    diffs = [mean(int(r["lazy_correct"] == "True") - int(r["fixed_correct"] == "True")
                  for r in group if r["problem"] == q) for q in ids]
    rng = random.Random(2025)
    b = sorted(sum(rng.choices(diffs, k=30))*100/30 for _ in range(50000))
    return dict(n=len(group), fixed=fixed, lazy=lazy, delta=(lazy-fixed)*100/len(group),
                ci=[b[1250], b[48749]], full=sum(r["full_switched"]=="True" for r in group),
                stops=[sum(r[a+"_finish"]=="length" for r in group) for a in ("fixed","lazy")],
                outcomes=dict(Counter("/".join(r[a+"_correct"] for a in ("fixed","lazy")) for r in group)))
for i in range(4):
    print("seed set",i,stats([r for r in rows if int(r["rep"])==i]))
print("new",stats([r for r in rows if int(r["rep"])>0]))
print("pooled",stats(rows))
