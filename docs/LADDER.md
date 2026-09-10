# Scaling ladder

Every rung is scored on the same held-out windows, reference trajectory, revisit pairs and matched starts, so rows are paired comparisons. Read every difference against the seed spread at the bottom, which the pre-registration defines as the ladder's resolution. Rules and their branches are in [LADDER_PREREG.md](LADDER_PREREG.md).

| run | params | ctx | tokens seen | epochs | val loss | cold acc | one-step moving | copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place (game) | still k=2-10 ident / churn | still k>10 ident |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder-2m-s0 | 2.0M | 6 | 11.38B | 15.61 | 1.920 | 0.408 | 27.33 dB | 21.20 | 30.73 | **64%** | +5.04 | +3.21 | +1.06 | -0.32 | 9.16 (11.25) dB | 79% / 11.8 | 100.0% |
| ladder-2m-t4-s0 | 2.0M | 6 | 2.92B | 4.00 | 2.130 | 0.359 | 26.61 dB | 21.20 | 30.73 | **57%** | +4.41 | +3.29 | +1.28 | +0.04 | 8.81 (11.25) dB | 87% / 2.2 | 100.0% |
| ladder-2m-t4-s1 | 2.0M | 6 | 2.92B | 4.00 | 2.128 | 0.363 | 26.71 dB | 21.20 | 30.73 | **58%** | +4.34 | +2.94 | +1.72 | -0.01 | 9.30 (11.25) dB | 92% / 4.2 | 100.0% |
| ladder-2m-t4-s2 | 2.0M | 6 | 2.80B | 3.83 | 2.144 | 0.356 | 26.65 dB | 21.20 | 30.73 | **57%** | +4.15 | +2.73 | +1.46 | +0.24 | 9.95 (11.25) dB | 97% / 7.0 | 100.0% |

Real-game hold-still reference: k=2-10 79.4% identical, 32.7 tokens when it moves; k>10 100%. Return-to-place is shown as model (game ceiling). Headroom is on the moving subset: (model - copy) / (ceiling - copy). Lead is model minus frozen-frame PSNR.

## Seed spread, the ladder's resolution

- **2M at T* (4 epochs), seeds 0-2** (3 runs): one-step moving PSNR 26.61 to 26.71 dB, **spread 0.10 dB**; headroom 57% to 58%, spread 1 points; val loss 2.128 to 2.144.

Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> --group <label>`; results accumulate in `docs/ladder_results.json`.
