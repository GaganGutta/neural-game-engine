# Scaling ladder

Every rung is scored on its final checkpoint, on the same held-out frames, reference trajectory, revisit pairs and hold-still starts, and every context length predicts the same target frames, so rows are paired comparisons. Read every difference against the seed spread at the bottom, which the pre-registration defines as the ladder's resolution. Rules and their branches are in [LADDER_PREREG.md](LADDER_PREREG.md).

| run | params | ctx | tokens seen | epochs | held-out loss | cold acc | one-step moving | copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place (game) | still k=2-10 ident / churn | still k>10 ident |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder-2m-t4-s0 | 2.0M | 6 | 2.92B | 4.00 | 2.158 | 0.345 | 27.07 dB | 21.77 | 30.97 | **58%** | +4.58 | +3.44 | +2.29 | +0.83 | 9.17 (12.02) dB | 87% / 2.2 | 100.0% |
| ladder-2m-t4-s1 | 2.0M | 6 | 2.92B | 4.00 | 2.151 | 0.347 | 27.12 dB | 21.77 | 30.97 | **58%** | +4.28 | +3.23 | +1.48 | +0.91 | 10.33 (12.02) dB | 92% / 4.2 | 100.0% |
| ladder-2m-t4-s2 | 2.0M | 6 | 2.92B | 4.00 | 2.172 | 0.341 | 27.04 dB | 21.77 | 30.97 | **57%** | +4.46 | +3.70 | +2.31 | +1.11 | 8.15 (12.02) dB | 94% / 4.0 | 100.0% |
| ladder-2m-s0 | 2.0M | 6 | 11.58B | 15.89 | 1.955 | 0.392 | 27.74 dB | 21.77 | 30.97 | **65%** | +5.28 | +4.08 | +1.27 | +1.51 | 11.00 (12.02) dB | 86% / 13.3 | 100.0% |
| ladder-8m-t4-s0 | 7.5M | 6 | 2.92B | 4.00 | 1.660 | 0.442 | 28.23 dB | 21.77 | 30.97 | **70%** | +5.64 | +4.17 | +2.48 | +1.52 | 9.02 (12.02) dB | 92% / 22.4 | 100.0% |
| ladder-26m-t4-s0 | 25.7M | 6 | 2.92B | 4.00 | 1.202 | 0.560 | 29.39 dB | 21.77 | 30.97 | **83%** | +6.77 | +6.46 | +5.81 | +2.56 | 9.60 (12.02) dB | 87% / 27.6 | 100.0% |
| ladder-ctx12-t4-s0 | 7.5M | 12 | 2.92B | 2.03 | 1.578 | 0.467 | 28.63 dB | 21.77 | 30.97 | **75%** | +5.93 | +5.19 | +4.01 | +0.77 | 12.22 (12.02) dB | 87% / 22.4 | 100.0% |
| ladder-ctx24-t4-s0 | 7.5M | 24 | 2.92B | 1.04 | 1.550 | 0.467 | 28.58 dB | 21.77 | 30.97 | **74%** | +6.29 | +5.73 | +4.44 | +0.88 | 8.68 (12.02) dB | 89% / 20.6 | 100.0% |

Real-game hold-still reference: k=2-10 79.4% identical, 32.7 tokens when it moves; k>10 100%. Return-to-place is shown as model (game ceiling) over 40 revisit pairs. Headroom is on the moving subset: (model - copy) / (ceiling - copy). Lead is model minus frozen-frame PSNR. Held-out loss is the training objective on 4096 fixed validation frames (`python -m ngx.eval.heldout`).

## Seed spread, the ladder's resolution

- **2M at T* (4 epochs), seeds 0-2** (3 runs): one-step moving PSNR 27.04 to 27.12 dB, **spread 0.08 dB**; headroom 57% to 58%, spread 1 points; held-out loss 2.151 to 2.172.

## 8M learning-rate probe

Each run is 0.4 epochs at batch 256 with the cosine planned over that budget; the winner is the lowest held-out loss at matched tokens, and an endpoint winner extends the probe by one point, as pre-registered. 26M uses the winner times sqrt(384/512).

| run | lr | tokens seen | held-out loss | cold loss | cold acc |
|---|---|---|---|---|---|
| ladder-8m-probe-lr0.5x | 0.00021 | 291.6M | 2.6347 | 3.0163 | 0.255 |
| ladder-8m-probe-lr1x | 0.00042 | 291.6M | 2.3853 | 2.7493 | 0.292 |
| ladder-8m-probe-lr2x | 0.00084 | 291.6M | 2.2236 | 2.5908 | 0.317 |
| ladder-8m-probe-lr4x **(winner)** | 0.00168 | 291.6M | 2.1245 | 2.4787 | 0.337 |

Winner 0.00168; 26M learning rate 0.00145.

Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> --group <label>` (and `--probe <run>=<lr> ...` for the probe table); results accumulate in `docs/ladder_results.json`.
