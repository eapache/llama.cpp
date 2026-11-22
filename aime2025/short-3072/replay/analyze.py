import csv
import sys
from pathlib import Path
import numpy as np
out = Path(__file__).resolve().parent
if len(sys.argv) == 2:
    out = out / sys.argv[1]
with (out/"drift.csv").open("w") as dst:
    writer = csv.writer(dst)
    writer.writerow(["trace","prefix_tokens","arm","kl_from_q8_nats","top1_match"])
    for src in sorted(out.glob("*-q8.f32")):
        trace = src.name.removesuffix("-q8.f32")
        positions = [int(row.split()[0]) for row in (out/f"{trace}-q8.bytes").read_text().splitlines()][1:]
        def logp(arm):
            values = np.fromfile(out/f"{trace}-{arm}.f32",dtype=np.float32).reshape(len(positions),-1).astype(np.float64)
            assert np.isfinite(values).all() and values.shape[1] >= 200000
            values -= values.max(axis=1,keepdims=True)
            return values - np.log(np.exp(values).sum(axis=1,keepdims=True))
        ref = logp("q8")
        for arm in ("fixed","lazy"):
            other = logp(arm)
            kl = (np.exp(ref)*(ref-other)).sum(axis=1)
            for i, value in enumerate(kl):
                writer.writerow([trace,positions[i],arm,value,int(ref[i].argmax()==other[i].argmax())])
