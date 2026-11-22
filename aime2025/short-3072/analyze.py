"""Read atomically completed trials; report paired differences by problem."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import json
import re
import random
import statistics

out = Path(__file__).resolve().parent
by = defaultdict(dict)
for path in sorted((out / "trials").glob("*.json.gz")):
    with gzip.open(path, "rt") as f:
        trial = json.load(f)
        s = trial["summary"]
        final = trial["response"]["choices"][0]["message"].get("content") or ""
        last_lines = "\n".join(final.split("\n")[-10:])
        boxes = re.findall(r"\\boxed{([^}]+)}", last_lines, re.I)
        nums = re.findall(r"\b(\d+)\b", last_lines)
        s["answer"] = ((boxes or nums or [""])[-1].strip() if s["finish"] == "stop" else "")
        s["correct"] = s["answer"] == s["gold"]
    by[s["id"], s["rep"]][s["arm"]] = s
ids = [x["id"] for x in json.loads((out.parent / "questions.json").read_text())]
pairs = [(p["fixed"],p["lazy"]) for p in by.values() if len(p)==2]
if not pairs:
    raise SystemExit("No completed pairs")
print(f"{len(pairs)} pairs, {sum(map(len,by.values()))} trials")
for i,label in enumerate(["Fixed Q4","Lazy Q8 to Q4"]):
    rows=[p[i] for p in pairs]
    print(label, "correct",sum(r["correct"] for r in rows),"length-limited",sum(r["finish"]=="length" for r in rows),"unanswered",sum(not r["answer"] for r in rows),"mean tokens",round(statistics.mean(r["tokens"] for r in rows)))
diffs = defaultdict(list)
for f,l in pairs:
    assert f["seed"]==l["seed"] and f["id"]==l["id"] and f["gold"]==l["gold"] and f["prompt_tokens"]==l["prompt_tokens"]
    diffs[f["id"]].append(int(l["correct"])-int(f["correct"]))
print("paired outcomes (fixed,lazy)",Counter((f["correct"],l["correct"]) for f,l in pairs))
print("switches (SWA,full)",Counter(tuple(any(int(t[2])==n for t in l["transitions"]) for n in (768,3072)) for f,l in pairs))
if len(pairs)!=30:
    raise SystemExit(0)
assert set(diffs)==set(ids) and all(len(v)==1 for v in diffs.values())
vals=[statistics.mean(diffs[i]) for i in ids]
rng=random.Random(2025)
boot=sorted(sum(rng.choices(vals,k=30))/30 for _ in range(50000))
print("difference pp",round(statistics.mean(vals)*100,2),"problem-bootstrap 95% CI",round(boot[1250]*100,2),round(boot[48749]*100,2))
with (out/"paired-results.csv").open("w") as file:
    w=csv.writer(file)
    w.writerow(["problem","rep","seed","gold","fixed_answer","lazy_answer","fixed_correct","lazy_correct","fixed_tokens","lazy_tokens","fixed_finish","lazy_finish","swa_switched","full_switched"])
    for f,l in sorted(pairs,key=lambda p:(ids.index(p[0]["id"]),p[0]["rep"])):
        w.writerow([f["id"],f["rep"],f["seed"],f["gold"],f["answer"],l["answer"],f["correct"],l["correct"],f["tokens"],l["tokens"],f["finish"],l["finish"],*[any(int(t[2])==n for t in l["transitions"]) for n in (768,3072)]])
