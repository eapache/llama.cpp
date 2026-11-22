# GPT-OSS-20B on AIME 2025

[Complete report on both context sizes, matched-prefix drift, and VRAM](report.md)

Lazy Q8_0 to Q4_0 improved exact-answer accuracy by 8.33 percentage points over fixed Q4_0. A percentile bootstrap resampling the 30 problems with all eight paired attempts together gave a 95% interval of +3.75 to +13.33 points (50,000 resamples, seed 2025).

| K and V cache | Correct | Accuracy | Mean completion tokens | Context-limit finishes |
| --- | ---: | ---: | ---: | ---: |
| Fixed Q4_0 | 65/240 | 27.08% | 2,496 | 1 |
| Lazy Q8_0 to Q4_0 | 85/240 | 35.42% | 2,220 | 1 |

Both arms solved 44 matched attempts. Lazy alone solved 41, fixed alone solved 21, and both missed 134. Across the eight seeds per problem, lazy scored higher on 14 problems, tied on 13, and scored lower on 3.

## Cache behavior

The full-attention cache started with 4,096 Q8_0 cells and could expand to 8,192 Q4_0 cells. The sliding-window cache started with 256 Q8_0 cells and could expand to 768 Q4_0 cells. Server logs confirm that 177 lazy attempts ended entirely in Q8_0, 36 switched only the sliding-window cache, 7 switched only full attention, and 20 switched both. This comparison measures the end-to-end cache policy. The accuracy gain is consistent with retaining higher precision early. Isolating drift after conversion would require a controlled comparison on matched prefixes.

## Reproduction

- Revision `85fe076f3752f8bac56098c356b87b7d9fad0ee2`, NVIDIA RTX PRO 4500 Blackwell, all model layers offloaded, Flash Attention on, default attention rotations enabled in both arms.
- Model `gpt-oss-20b-MXFP4.gguf`. The weight hash is in [model.sha256](model.sha256).
- AIME 2025 I and II, all 30 questions, eight attempts per question per arm. Prompts and normalized gold answers were checked against the linked Q4_0 evaluation artifact. Sampling used low reasoning effort, temperature 1, top-k 0, top-p 1, min-p 0.01. [Reference evaluation](https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4150413357)
- A fresh single-slot server per attempt, 8,192 context tokens, context shifting off. The arms used identical seeds for each problem and repetition, and their order alternated. Each recorded request includes the sampling parameters.
- Scoring used the last ten lines of the final answer, prioritizing the last boxed answer with fallback to the last integer. Non-stop finishes and missing answers count as incorrect. The published reference used separate model samples; its rotated Q4_0 score was 21.7%.
- [Run script](run.py), [scoring and bootstrap](analyze.py), [paired results](paired-results.csv), and [per-trial requests, responses and logs](trials/). Run `python3 analyze.py` to regenerate the paired CSV from the 480 saved trials.
