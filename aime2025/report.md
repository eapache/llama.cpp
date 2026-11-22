# Does lazy KV quantization improve GPT-OSS-20B on AIME 2025?

Lazy Q8-to-Q4 improved accuracy in the 8,192-token test (+8.33 percentage points). The smaller context made full-cache conversion nearly seven times as frequent, and the pooled four-seed short-context result favored fixed Q4 by 3.33 percentage points (31/120 fixed, 27/120 lazy). On identical reasoning prefixes, lazy stayed closer to fixed Q8 on average before the full cache converted. That advantage did not persist on average afterward. The two policies allocated exactly the same KV bytes at each context size. Whole-process VRAM differed by a few MiB.

## Protocol

Both experiments ran the same local GPT-OSS-20B MXFP4 weights and the same llama.cpp binary (revision `85fe076f3752f8bac56098c356b87b7d9fad0ee2`) on one NVIDIA RTX PRO 4500 Blackwell. The fixed arm used `-ctk q4_0 -ctv q4_0`. The lazy arm used `-ctk q8_0:q4_0 -ctv q8_0:q4_0`. Full-attention and sliding-window caches can switch independently. In lazy mode, the initial Q8 rows occupy the Q4 allocation. On conversion, the cache requantizes occupied rows in place and uses Q4 for subsequent rows. [Model checksum](model.sha256) · [Original benchmark discussion](https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4150413357)

Each test covered all 15 problems from AIME I and all 15 from AIME II. Each pair shared a question, prompt and seed. Arm order alternated through a shuffled problem order. Every attempt started a fresh single-slot server with all model layers on CUDA0, Flash Attention and attention rotations enabled, warmup off, prompt caching off, and context shifting off. Both used low reasoning effort, temperature 1, top-k 0, top-p 1 and min-p 0.01. The response-token cap matched the context; the prompt used part of that budget. The grader examined the last ten lines of the final response, took the last boxed answer or otherwise the last integer, and required an exact match to the normalized gold answer. Missing answers and length stops scored zero. [Questions and answers](questions.json) · [Long-run procedure](run.py) · [Short-run procedure](short-3072/run.py)

The original 8,192-token run used eight seeds per question (240 pairs, 480 generations). The follow-up used 3,072 tokens and one fresh seed per question (30 pairs, 60 generations). Seeds were `2025000 + 30 * repetition + problem_index` and `2026000 + problem_index`, respectively, with indices starting at zero. The smaller context lowers the full-cache switch from 4,096 to 1,536 rows. The sliding-window cache starts at 256 Q8 cells and expands to 768 Q4 cells in both runs. The follow-up trades seed coverage and answer space for more frequent conversion. Each accuracy interval below resamples whole problems with replacement, keeping the arm pairing and all seeds for each selected problem (50,000 draws, seed 2025).

## Four seeds per problem at 3,072 tokens

The three additional samples brought the short-context estimate closer to zero. Lazy trailed by one correct answer across the 90 new pairs. Across all four seeds, its accuracy gap narrowed from -10.00 to -3.33 percentage points. The 95% problem-bootstrap interval for the pooled difference was -11.67 to +4.17 points. These data leave room for a modest benefit or a larger loss at this context size.

| Paired seed set | Fixed Q4 correct | Lazy correct | Lazy minus fixed | Lazy full-cache conversions | Length stops, fixed / lazy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 9/30 | 6/30 | -10.00 pp | 23/30 | 10 / 10 |
| Additional 1 | 7/30 | 7/30 | 0.00 pp | 18/30 | 10 / 9 |
| Additional 2 | 7/30 | 8/30 | +3.33 pp | 22/30 | 7 / 11 |
| Additional 3 | 8/30 | 6/30 | -6.67 pp | 22/30 | 8 / 10 |
| Three additional | 22/90 (24.44%) | 21/90 (23.33%) | -1.11 pp | 62/90 | 25 / 30 |
| All four | 31/120 (25.83%) | 27/120 (22.50%) | -3.33 pp | 85/120 | 35 / 40 |

Across all 120 matched attempts, both arms solved 20, lazy alone solved seven, fixed alone solved 11, and both missed 82. The full-attention cache converted in 70.8% of lazy attempts. Length stops counted as incorrect in both arms. These accuracy runs measure end-to-end behavior, including truncation. The matched-prefix replay below addresses drift separately.

The additional runs used repetitions 1 through 3 of `2026000 + 30 * repetition + problem_index`. All 180 requests matched the original short-run requests for their respective problem and arm apart from the seed. The server command lines matched apart from the ephemeral port. Run `python3 short-extra/run.py`, `python3 short-extra/analyze.py`, and `python3 short-extra/pool.py` from the report directory to reproduce the extension and pooled analysis. [New trials and server logs](short-extra/trials/) · [Three new seed sets](short-extra/paired-results.csv) · [All four seed sets](short-extra/pooled-results.csv)

## Original accuracy and cache transitions

| Context | Correct, fixed Q4 | Correct, lazy | Lazy minus fixed (95% interval) | Mean output tokens, fixed / lazy | Length stops, fixed / lazy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 65/240 (27.08%) | 85/240 (35.42%) | +8.33 pp (+3.75, +13.33) | 2,496 / 2,220 | 1 / 1 |
| 3,072 | 9/30 (30.00%) | 6/30 (20.00%) | -10.00 pp (-23.33, +0.00) | 2,010 / 2,073 | 10 / 10 |

| Context | Both correct | Lazy only | Fixed only | Both wrong | Neither cache switched | SWA only | Full only | Both switched |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 44 | 41 | 21 | 134 | 177 | 36 | 7 | 20 |
| 3,072 | 6 | 0 | 3 | 21 | 6 | 1 | 17 | 6 |

At 8,192 tokens, lazy gained 20 correct answers over 240 paired attempts. Across the eight seeds it won on 14 problems, tied on 13 and lost on three. Only 27/240 lazy attempts (11.25%) switched the full-attention cache, and 177/240 stayed in Q8 throughout. The longer run chiefly measures the benefit of the whole policy, including attempts that finish before conversion.

At 3,072 tokens, 23/30 (76.7%) switched the full cache and 24/30 switched at least one cache. Fixed alone solved I-02, I-04 and II-07; all three lazy partners crossed the full-cache threshold. Of the 23 pairs whose lazy run crossed it, fixed solved five and lazy solved two. That subgroup is selected by the generated trajectory. It cannot isolate the causal effect of conversion. Only three pairs differed in correctness across the whole follow-up (two-sided exact paired sign test, p = 0.25). The observed short-run result favors fixed Q4, with substantial uncertainty from one seed per question. Ten length stops per arm further limit what this comparison says about longer solutions. Changing both the context and seeds between experiments also precludes attributing the difference between their accuracy estimates to context alone.

## Drift on identical token histories

To separate changes in predicted probabilities from changes in the sampled text, I replayed the same tokens under fixed Q8, fixed Q4 and lazy Q8-to-Q4. I selected the first two problems from each AIME paper whose original fixed-Q4 seed-0 completion contained at least 2,900 tokens, using length alone (I-10, I-11, II-03 and II-06). I reconstructed a low-effort Harmony prompt, appended the saved reasoning text, tokenized once per run with the same model vocabulary, and teacher-forced the first 2,560 tokens. At each checkpoint I compared the full next-token distribution against fixed Q8 using KL divergence in nats, before sampling or filtering. Lower KL means closer to fixed Q8. Top-1 agreement counts checkpoints where the most probable token matched.

I used two prefill schedules so the comparison after conversion could distinguish a sliding-window cache still in Q8 from one already in Q4. With chunks of 128, the sliding-window cache stayed Q8 throughout. With an initial chunk of 384 followed by chunks of 128, it switched to Q4 before the first chunk. The logs confirm full-cache conversion through row 1,536 on all four traces in both schedules. Checkpoints occur at the end of each chunk, so the first post-conversion measurement is at 1,664 tokens. Each row below compares identical prefixes and chunk sizes across all three arms.

| Sliding-window cache in lazy arm | Full-cache phase | Checkpoints | Mean KL, fixed Q4 | Mean KL, lazy | Top-1 matches, fixed / lazy |
| --- | --- | ---: | ---: | ---: | ---: |
| Stays Q8 | Before conversion | 48 | 0.25823 | 0.02501 | 41 / 48 |
| Stays Q8 | After conversion | 32 | 0.07296 | 0.10386 | 28 / 28 |
| Switched to Q4 | Before conversion | 40 | 0.10457 | 0.07852 | 37 / 37 |
| Switched to Q4 | After conversion | 32 | 0.12408 | 0.12639 | 28 / 27 |

Once both caches were Q4, lazy had lower mean KL on two traces and higher mean KL on two. Its overall post-conversion mean differed from fixed Q4 by just +0.00231 nats. With the sliding-window cache still in Q8, lazy had higher post-conversion mean KL on three of four traces. These four traces show an early average precision benefit and no consistent post-conversion advantage. The checkpoints within each trace are correlated. Fixed Q8 is a numerical reference, and teacher forcing measures local predictive drift on a shared path. The AIME scores above measure what happens when each arm generates its own path. [Preparation and trace selection](short-3072/replay/prepare.py) · [128-token schedule](short-3072/replay/drift.csv) · [Both caches converted](short-3072/replay/both-converted/drift.csv)

## GPU memory

**The two policies reserve exactly the same number of CUDA bytes for KV at each context size. Whole-process VRAM differs.** The probe reads distinct KV backing buffers with `ggml_backend_buffer_get_size` after decoding. It also reads the model, context and compute categories from `llama_get_memory_breakdown`. Context includes the two rotation buffers, together 786,432 bytes. The probe matches the server's 2,048 batch, 512 microbatch, single output, compact sliding-window cache, offload and Flash Attention settings. The process columns come independently from NVML on the actual benchmark server.

| Context | Cache policy | CUDA KV buffers, bytes | CUDA KV + rotation, bytes | CUDA model, bytes | CUDA startup compute, bytes | Server VRAM at readiness, MiB | Server peak VRAM, MiB (median [range]) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | Fixed Q4 | 61,931,520 | 62,717,952 | 11,481,256,960 | 103,605,376 | 11402 | 11504 [11504-11504] |
| 8,192 | Lazy | 61,931,520 | 62,717,952 | 11,481,256,960 | 98,868,352 | 11398 | 11500 [11500-11500] |
| 3,072 | Fixed Q4 | 26,542,080 | 27,328,512 | 11,481,256,960 | 98,342,016 | 11364 | 11464 [11462-11466] |
| 3,072 | Lazy | 26,542,080 | 27,328,512 | 11,481,256,960 | 96,236,672 | 11362 | 11464 [11460-11466] |

The short-context KV allocation stayed at 26,542,080 bytes at every replay checkpoint, including before and after full-cache conversion. At server readiness, lazy used 2 MiB less in all 30 short pairs. The sampled peaks matched in 24 pairs and lazy peaked 2 MiB lower in six. The long-context process readings are a separate repeat of problem I-10, seed 2025009, one server per arm. The original 480 runs recorded allocator logs but did not sample NVML. In that repeat the lazy arm switched the sliding-window cache only, and used 4 MiB less both at readiness and at sampled peak. Compute buffers vary with cache representation and graph reservation. These byte counts and process readings apply to this model, backend and build.

## Reproducing the measurements

The data, scripts and report are in the local `build-aime2025/` directory. Paths below are relative to that directory except for the probe build and replay commands, which are relative to the checkout root.

- For the original run, use `python3 run.py` and `python3 analyze.py` from this directory. [Paired scores](paired-results.csv) · [480 requests, responses and server logs](trials/)
- For the shorter run, use `python3 short-3072/run.py` and `python3 short-3072/analyze.py`. Both runners resume completed trials. [Paired scores](short-3072/paired-results.csv) · [60 requests, responses, VRAM samples summarized per request, and server logs](short-3072/trials/)
- The long-context VRAM spot check uses `python3 long-memory/run.py`. The sampler queries the compute-process entry for the server PID on device zero at readiness, approximately every 100 ms during the request, and at completion. It records bytes reported by NVML. Peaks are sampled lower bounds. [Sampler](short-3072/mem.py) · [Long-context spot check](long-memory/trials/)
- To reproduce the matched-prefix replay from the checkout root, compile the probe with the command below, then run `python3 build-aime2025/short-3072/replay/prepare.py`, `bash build-aime2025/short-3072/replay/run.sh`, and `python3 build-aime2025/short-3072/replay/analyze.py`. For the second schedule, run `bash build-aime2025/short-3072/replay/both-converted/run.sh` and `python3 build-aime2025/short-3072/replay/analyze.py both-converted`. [Probe](short-3072/replay/probe.cpp) · [Per-prefix scores](short-3072/replay/drift.csv) · [CUDA byte counts and conversion logs](short-3072/replay/)

```sh
g++ -std=c++17 -Iinclude -Iggml/include -Isrc \
    build-aime2025/short-3072/replay/probe.cpp \
    -Lbuild/bin -Wl,-rpath,"$PWD/build/bin" -lllama -lggml -lggml-base \
    -o build-aime2025/short-3072/replay/probe
```
