# Cache carrying: latency versus exactness

Weights: untrained (random init, config shape). 6 matched starts (replayed prefix in VizDoom, then a fixed random action plan), 60 frames each, greedy decoding, 3 warmup frames excluded from timing. Device `cpu`.

Every regime starts from the same context and applies the same actions. The reference is a full recompute of the prefix cache on every frame, which is exact. Divergence is measured against that rollout frame by frame; once two rollouts differ, the difference feeds back through the context, so it compounds and is reported per k as well as on average.

| regime | ms/frame | vs. full | mean PSNR vs ref | mean tokens differing | frames identical | first divergence (frame, per start) |
|---|---|---|---|---|---|---|
| full recompute every frame | 40.74 | 1.00x | identical | 0.0 / 64 | 100.0% | never, never, never, never, never, never |
| carry, refresh every 4 | 35.20 | 1.16x | 70.7 dB | 19.4 / 64 | 63.3% | 7, 7, 7, 7, 7, 7 |
| carry, refresh every 8 | 33.92 | 1.20x | 63.5 dB | 25.5 / 64 | 55.0% | 5, 5, 5, 5, 7, 5 |
| carry, refresh every 16 | 33.13 | 1.23x | 41.2 dB | 42.2 / 64 | 28.1% | 5, 5, 5, 5, 7, 5 |
| carry, never refreshed | 32.65 | 1.25x | 50.8 dB | 35.3 / 64 | 39.7% | 5, 5, 5, 5, 7, 5 |

Per-k divergence from the full-recompute rollout, mean over starts. PSNR is clipped at 100 dB where frames are byte-identical.

| regime | k=1 | k=8 | k=16 | k=32 | k=60 |
|---|---|---|---|---|---|
| carry, refresh every 4 | 100.0 dB / 0.0 tok | 27.5 dB / 23.3 tok | 11.4 dB / 64.0 tok | 86.1 dB / 10.7 tok | 100.0 dB / 0.0 tok |
| carry, refresh every 8 | 100.0 dB / 0.0 tok | 26.3 dB / 31.2 tok | 9.8 dB / 64.0 tok | 86.9 dB / 10.7 tok | 100.0 dB / 0.0 tok |
| carry, refresh every 16 | 100.0 dB / 0.0 tok | 26.3 dB / 31.2 tok | 14.4 dB / 64.0 tok | 29.0 dB / 53.3 tok | 56.4 dB / 32.0 tok |
| carry, never refreshed | 100.0 dB / 0.0 tok | 26.3 dB / 31.2 tok | 14.4 dB / 64.0 tok | 57.8 dB / 32.0 tok | 73.4 dB / 21.3 tok |

## The approximation itself, measured without compounding

A shadow carried cache is extended along the exact rollout's own windows and never allowed to steer, so at every frame it decodes the same window the exact cache does. `max dlogit` is the first-pass logit perturbation in units of the logit standard deviation (how close it came to flipping a decision); `tokens differ` is whether the full MaskGIT decode actually came out different on that frame.

| eviction depth | max dlogit (sd), mean over starts | worst start | 4-pass tokens differ | starts affected |
|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.0 / 64 | 0% |
| 2 | 0.044 | 0.065 | 0.0 / 64 | 0% |
| 3 | 0.036 | 0.057 | 0.0 / 64 | 0% |
| 4 | 0.058 | 0.073 | 0.0 / 64 | 0% |
| 5 | 0.129 | 0.197 | 17.3 / 64 | 83% |
| 6 | 0.263 | 0.297 | 0.8 / 64 | 17% |
| 8 | 0.521 | 0.546 | 44.5 / 64 | 100% |
| 12 | 0.614 | 0.716 | 31.8 / 64 | 100% |
| 16 | 0.386 | 0.449 | 38.7 / 64 | 67% |
| 24 | 0.241 | 0.300 | 0.0 / 64 | 0% |

Depth 1 is exact by construction. The perturbation grows with depth for the first several frames as each retained block inherits history a fresh recompute would not give it, and a refresh every K frames caps the depth at K-1. Once a single token differs, the rollouts have different context and nothing recovers them, so a refresh bounds the *size* of the approximation and not the divergence of the rollout.


**No carrying regime is exact on this model.** The table gives the trade; the never-refreshed regime differs on 60.3% of frames by 35.3 tokens on average for 1.25x. Whether that is acceptable is a judgement about the demo, and it is stated here as a number rather than made silently.

**These divergence numbers describe an untrained model, and untrained models are not a usable proxy in either direction.** Two random initialisations of this exact shape were checked: one never diverged in twelve frames, the other diverged at frame 5 on every start. Sensitivity to a fixed-size perturbation depends on how close the model's decisions sit to a tie, and random init puts that anywhere. Latency does not depend on weights and stands. Exactness is decided on the first trained rope checkpoint, and until then the engine defaults to the exact path.

Regenerate with `python -m ngx.eval.cache_cadence --config configs/small.yaml --untrained`.
