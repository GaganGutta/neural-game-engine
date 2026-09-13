# Scaling ladder

Every rung is scored on its final checkpoint, on the same held-out frames, reference trajectory, revisit pairs and hold-still starts, and every context length predicts the same target frames, so rows are paired comparisons. Read every difference against the seed spread at the bottom, which the pre-registration defines as the ladder's resolution. Rules and their branches are in [LADDER_PREREG.md](LADDER_PREREG.md).

| run | params | ctx | tokens seen | epochs | held-out loss | cold acc | one-step moving | copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place (game) | still k=2-10 ident / churn | still k>10 ident |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder-2m-t4-s0 | 2.0M | 6 | 2.92B | 4.00 | 2.158 | 0.345 | 27.07 dB | 21.77 | 30.97 | **58%** | +4.58 | +3.44 | +2.29 | +0.83 | 9.17 (12.02) dB | 87% / 2.2 | 100.0% |
| ladder-2m-t4-s1 | 2.0M | 6 | 2.92B | 4.00 | 2.151 | 0.347 | 27.12 dB | 21.77 | 30.97 | **58%** | +4.28 | +3.23 | +1.48 | +0.91 | 10.33 (12.02) dB | 92% / 4.2 | 100.0% |
| ladder-2m-t4-s2 | 2.0M | 6 | 2.92B | 4.00 | 2.172 | 0.341 | 27.04 dB | 21.77 | 30.97 | **57%** | +4.46 | +3.70 | +2.31 | +1.11 | 8.15 (12.02) dB | 94% / 4.0 | 100.0% |
| ladder-2m-s0 | 2.0M | 6 | 11.58B | 15.89 | 1.955 | 0.392 | 27.74 dB | 21.77 | 30.97 | **65%** | +5.28 | +4.08 | +1.27 | +1.51 | 11.00 (12.02) dB | 86% / 13.3 | 100.0% |

Real-game hold-still reference: k=2-10 79.4% identical, 32.7 tokens when it moves; k>10 100%. Return-to-place is shown as model (game ceiling) over 40 revisit pairs. Headroom is on the moving subset: (model - copy) / (ceiling - copy). Lead is model minus frozen-frame PSNR. Held-out loss is the training objective on 4096 fixed validation frames (`python -m ngx.eval.heldout`).

## Seed spread, the ladder's resolution

- **2M at T* (4 epochs), seeds 0-2** (3 runs): one-step moving PSNR 27.04 to 27.12 dB, **spread 0.08 dB**; headroom 57% to 58%, spread 1 points; held-out loss 2.151 to 2.172.

Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> --group <label>`; results accumulate in `docs/ladder_results.json`.
