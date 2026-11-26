# CPU distribution sampler performance

The prototype makes two changes in `llama_sampler_dist_apply`. For unsorted candidates, it scans even and odd positions with independent maximum-logit accumulators, then combines them. For normalization, it computes one double-precision reciprocal and multiplies each probability by it. Both changes apply to ordinary CPU sampling and speculative decoding.

Standalone CPU sampler time for 200,058 unsorted candidates on an AMD Ryzen 9 9900X, GCC 15.2 release build.

| Implementation | Time per call |
|---|---:|
| Stock | 0.96 ms |
| Reciprocal normalization alone | 0.90 ms |
| Reciprocal plus two-way maximum scan | 0.55 ms |

The two-way scan cut sampler time by about 39% relative to reciprocal normalization alone. Together the changes cut it by about 43% relative to stock. With already-sorted candidates the maximum scan is skipped. Normalization alone reduced time from 0.57 to 0.48 ms at the same vocabulary size.

The two-way scan is independently useful. An open PR already implements reciprocal normalization and retains a single-accumulator maximum scan. [PR #24934](https://github.com/ggml-org/llama.cpp/pull/24934)

## Wide-distribution speculative verification

On a rejected draft token, the verifier samples from weights proportional to `max(0, p - q)`, where `p` is the target distribution and `q` is the draft distribution. The prototype skips sorting the target candidates and indexes draft probabilities by token ID, reducing the residual calculation to linear time for a full vocabulary. When both candidate lists have at most 64 entries, it uses a short linear lookup to avoid allocating a vocabulary-sized buffer.

With three probabilistic draft tokens, the verifier change alone cut mean verification time from 21.67 to 2.50 ms per round on Inkling and from 30.43 to 2.92 ms on full-vocabulary Qwen. The table gives pooled generation throughput on one prompt.

| Model and target sampling | Stock | Verifier change alone | Verifier and CPU sampler changes |
|---|---:|---:|---:|
| Inkling-2-Mini DSpark, full vocabulary | 45.8 tok/s | 87.9 tok/s | 91.2 tok/s |
| Qwen3.8-27B MTP, full vocabulary | 38.5 tok/s | 58.7 tok/s | 60.8 tok/s |
| Qwen3.8-27B MTP, top-20 preset | 73.5 tok/s | 73.8 tok/s | 73.4 tok/s |

The full-vocabulary Qwen comparison used two seeds and the other comparisons used four. Skipping the sort preserves the residual distribution but can change fixed-seed output by changing the order in which weights are sampled. [Per-seed timings](build/inkling-dspark-prototypes/adaptive-verifier-results.txt)
