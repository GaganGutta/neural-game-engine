# B0: baselines

Model: 2.0M params, context 6 frames, val loss 3.203, cold token accuracy 0.152. Greedy decoding, 4-pass MaskGIT, memory off.

## One-step, teacher-forced

Real frames for context, one frame predicted, 1600 held-out windows. This is the test that cannot be excused by drift.

Split by whether the two real frames are pixel-identical. On those, copy-last-frame is exactly right and PSNR is infinite, so any aggregate that mixes them is decided by where the infinity is clipped rather than by the models. Splitting removes the clipping choice from the comparison instead of managing it.

### Moving transitions (1555 windows, 97.2%)

No infinities here, so no cap exists and the mean is well defined.

| predictor | mean PSNR | median PSNR | token accuracy |
|---|---|---|---|
| copy-last-frame | 21.05 dB | 21.76 dB | 0.097 |
| model (2.0M), greedy | **23.06 dB** | 23.85 dB | 0.152 |
| model (2.0M), sampled | 23.05 dB | 23.92 dB | 0.125 |
| tokenizer ceiling | 31.04 dB | 30.80 dB | 1.000 |

**The model beats copy-last-frame by 2.00 dB on moving transitions.** Headroom from copy to the tokenizer ceiling is 9.99 dB, so it captures **20% of what was available**. Token accuracy agrees and is cap-free by construction.

### Static transitions (45 windows, 2.8%)

Frames where nothing moved: a no-op action, or the agent pressed against a wall.

| predictor | mean PSNR | median PSNR | token accuracy |
|---|---|---|---|
| copy-last-frame | exact (infinite) | exact | 1.000 |
| model (2.0M), greedy | 28.80 dB | 28.97 dB | 0.723 |
| tokenizer ceiling | 29.91 dB | | 1.000 |

**The model does not beat copy-last-frame here, and cannot.** Copy is exact by construction; the model scores 28.80 dB. It reproduces the previous frame's tokens exactly on **0%** of static transitions.

This is the number that predicts a demo-visible artifact. When the player stands still, the world should be frozen. Every static transition where the model emits different tokens is a frame that changes when it should not, which reads as shimmer. Note the ceiling is finite here too: anything that round-trips through the codebook cannot be pixel-exact, so perfect stillness is only reachable by emitting *identical tokens*, not by predicting well. That makes token-repeat rate, not PSNR, the metric to watch for this artifact.

## Closed-loop

32 independent rollouts of 32 frames from held-out starts, averaged per step. The model consumes its own predictions; copy-last-frame degenerates to freezing on the last real frame. Reported per k and never averaged over k, because the trajectory is not monotonic and a single mean over it would describe nothing. PSNR is clipped at 100 dB in this section only, for the rare case of a rollout that starts from a stalled agent.

| k | copy (frozen) | model | tokenizer ceiling | model lead |
|---|---|---|---|---|
| 1 | 21.02 dB | **23.67 dB** | 30.80 dB | +2.65 dB |
| 2 | 20.37 dB | **22.26 dB** | 30.91 dB | +1.89 dB |
| 4 | 19.36 dB | **21.27 dB** | 31.14 dB | +1.91 dB |
| 8 | 17.91 dB | **19.36 dB** | 31.20 dB | +1.45 dB |
| 16 | 17.58 dB | **17.97 dB** | 31.01 dB | +0.39 dB |
| 32 | 15.45 dB | **15.60 dB** | 31.35 dB | +0.15 dB |

The model's lead over a frozen frame decays from +2.65 dB at k=1 to +0.15 dB at k=32. Past roughly k=16 it is not meaningfully better than showing the player a still image, which is the honest way to read the rollout GIF.

The tokenizer ceiling is flat across k, as it must be: it re-encodes the true frame at every step and never compounds.

Regenerate with `python -m ngx.eval.baselines --config configs/small.yaml`.
