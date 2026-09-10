# Scaling ladder

Every rung is scored on the same held-out windows, reference trajectory, revisit pairs and matched starts, so rows are paired comparisons. Read every difference against the seed spread at the bottom, which the pre-registration defines as the ladder's resolution. Rules and their branches are in [LADDER_PREREG.md](LADDER_PREREG.md).

| run | params | ctx | tokens seen | epochs | val loss | cold acc | one-step moving | copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place (game) | still k=2-10 ident / churn | still k>10 ident |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ladder-2m-s0 | 2.0M | 6 | 11.38B | 15.61 | 1.920 | 0.408 | 27.33 dB | 21.20 | 30.73 | **64%** | +5.04 | +3.21 | +1.06 | -0.32 | 9.16 (11.25) dB | 79% / 11.8 | 100.0% |

Real-game hold-still reference: k=2-10 79.4% identical, 32.7 tokens when it moves; k>10 100%. Return-to-place is shown as model (game ceiling). Headroom is on the moving subset: (model - copy) / (ceiling - copy). Lead is model minus frozen-frame PSNR.

## Seed spread, the ladder's resolution


Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> --group <label>`; results accumulate in `docs/ladder_results.json`.
