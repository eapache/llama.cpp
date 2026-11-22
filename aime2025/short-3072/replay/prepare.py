from pathlib import Path
import gzip
import json

out = Path(__file__).resolve().parent
system = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\nCurrent date: 2026-11-22\n\nReasoning: low\n\n"
          "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
for split in ("I", "II"):
    picked = 0
    for path in sorted((out.parents[1]/"trials").glob(f"00-{split}-*-fixed.json.gz")):
        trial = json.load(gzip.open(path))
        if trial["summary"]["tokens"] < 2900:
            continue
        text = (system + "<|start|>user<|message|>" + trial["request"]["messages"][0]["content"] +
                "<|end|><|start|>assistant<|channel|>analysis<|message|>" +
                trial["response"]["choices"][0]["message"]["reasoning_content"])
        (out/(trial["summary"]["id"]+".txt")).write_text(text)
        picked += 1
        if picked == 2:
            break
    assert picked == 2
