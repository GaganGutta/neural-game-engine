# B0: baselines

Model: 2.0M params, context 6 frames, val loss 3.203, cold token accuracy 0.152. Greedy decoding, 4-pass MaskGIT, memory off.

## One-step, teacher-forced

Real frames for context, one frame predicted, 200 held-out windows. This is the test that cannot be excused by drift.

| predictor | PSNR median | PSNR mean (cap-dependent) | token accuracy | beats copy on |
|---|---|---|---|---|
| copy-last-frame | 22.11 dB | 23.85 dB | 0.129 | -- |
| model (2.0M), sampled | 23.65 dB | 23.42 dB | 0.140 | 80% of windows |
| model (2.0M), greedy | **23.51 dB** | 23.49 dB | **0.177** | **80% of windows** |
| tokenizer ceiling | 31.00 dB | 31.31 dB | 1.000 by construction | -- |

**The model beats copy-last-frame by +1.40 dB on the median**, wins on 80% of individual windows, and sits 7.49 dB below the tokenizer ceiling.

Read the median and the win rate, not the mean. 3.5% of consecutive frame pairs here are pixel-identical (a no-op action, or the agent pressed against a wall), and copy-last-frame is exactly right on every one of them. PSNR is infinite on those, so the mean depends entirely on where the infinity is clipped: at the 100 dB cap used here copy means 23.85 dB, and moving the cap moves that by roughly the identical fraction times the change. The median is unaffected while identical pairs stay a minority, and the win rate compares two numbers per window and cannot be clipped at all.

Headroom, measured on medians: the gap from copy-last-frame to the tokenizer ceiling is 8.89 dB, and the model captures 1.40 dB of it, or **16% of what was available**. Token accuracy is cap-free and agrees: 0.129 for copy against 0.177 for the model.

Greedy and sampled decoding land 0.07 dB apart, which is nothing against the 7.5 dB frame-to-frame spread. Decoder temperature is not a meaningful lever at this model size.

## Closed-loop

32 independent rollouts of 32 frames from held-out starts, averaged per step. The model consumes its own predictions; copy-last-frame degenerates to freezing on the last real frame. Reported per k and never averaged over k, because the trajectory is not monotonic and a single mean over it would describe nothing.

| k | copy (frozen) | model | tokenizer ceiling | model lead |
|---|---|---|---|---|
| 1 | 21.02 dB | **23.67 dB** | 30.80 dB | +2.65 dB |
| 2 | 20.37 dB | **22.26 dB** | 30.91 dB | +1.89 dB |
| 4 | 19.36 dB | **21.27 dB** | 31.14 dB | +1.91 dB |
| 8 | 17.91 dB | **19.36 dB** | 31.20 dB | +1.45 dB |
| 16 | 17.58 dB | **17.97 dB** | 31.01 dB | +0.39 dB |
| 32 | 15.45 dB | **15.60 dB** | 31.35 dB | +0.15 dB |

The model's lead over a frozen frame decays from +2.65 dB at k=1 to +0.15 dB at k=32. Past roughly k=16 it is not meaningfully better than showing the player a still image, which is the honest way to read the rollout GIF.

The tokenizer ceiling is flat across k, as it must be: it re-encodes the true frame at every step and never compounds. It is drawn here as the horizontal line everything else is failing to reach.

Regenerate with `python -m ngx.eval.baselines --config configs/small.yaml`.
