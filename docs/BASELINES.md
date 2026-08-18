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

### Static transitions (n = 45 windows, 2.8% of the sample)

Frames where nothing moved: a no-op action, or the agent pressed against a wall.

| predictor | mean PSNR | median PSNR | token accuracy |
|---|---|---|---|
| copy-last-frame | exact (infinite) | exact | 1.000 |
| model (2.0M), greedy | 28.80 dB | 28.97 dB | 0.723 |
| tokenizer ceiling | 29.91 dB | | 1.000 |

**The model does not beat copy-last-frame here, and cannot.** Copy is exact by construction; the model scores 28.80 dB. It reproduces the previous frame's tokens exactly on **0%** of static transitions (n = 45).

Read those two numbers differently. At n = 45 the 0% exact-repeat result is decisive: not one static frame in 45 came back unchanged. The 28.80 dB mean is not; it is a small sample and should be treated as indicative only.

### Why stillness is a hard constraint, not a quality problem

A still frame requires **all 64 tokens correct simultaneously**. Treating per-token accuracy as roughly independent, the probability of an unchanged frame is that accuracy raised to the 64th power:

| per-token agreement | P(frame unchanged) |
|---|---|
| 0.723 (this model) | ~0.000 |
| 0.99 | 0.52 |
| 0.995 | 0.73 |
| 0.999 | 0.94 |

So even **99% per-token agreement flickers on about half of all still frames**. Reliable stillness needs roughly 99.9%, and that is not a quality target that cross-entropy has any incentive to reach. The loss is a mean over tokens: getting 46 of 64 right is a perfectly acceptable value, and it is also a visibly broken frame. The objective and the artifact are measuring different things, and no amount of lowering the loss changes the exponent.

This is why the problem is unlikely to yield to capacity. Going from 0.723 to 0.99 per-token would be an enormous modelling win and would still leave the world shimmering half the time it should be frozen.

### Churn under a held no-op (472 closed-loop frame pairs)

The static row above is teacher-forced. This is the same question asked the way a viewer meets it: the model consuming its own output for 60 frames while the player holds the no-op action.

| measure | value |
|---|---|
| mean tokens changed per frame | **0.7 / 64** |
| frames byte-identical to the previous one | **90.5%** |

**This is much better than the teacher-forced static row predicts, and the discrepancy is the interesting part.** Teacher-forced, the model reproduces a real previous frame's tokens exactly 0% of the time. Closed-loop, it reproduces its own previous tokens 90.5% of the time. Those are different tasks: matching an external frame exactly is hard, while settling on a self-consistent fixed point is something an autoregressive model falls into naturally, because its context is already full of the frame it just produced.

So the practical artifact is milder than the static row alone suggests. Roughly 10% of held-still frames change at all, and those that do change about 8 of 64 patches. That reads as an occasional twitch in a small region rather than continuous shimmer over the whole screen. Worth stating plainly: the earlier prediction of visible shimmer was drawn from the teacher-forced number, and the closed-loop measurement walks it back.

The combinatorial argument above is unaffected and still explains why *exact* stillness is not something the objective can be pushed into. It is simply that self-consistency, not accuracy, is doing the work here, and self-consistency is not a property the loss is optimising either. It could get worse with a stronger model that tracks the world more sharply instead of settling, which is a reason to re-run this measurement on every rung of the scaling ladder rather than assume it holds.

These are the before-numbers for any later fix.

### Candidate fix, logged and not implemented

A per-token **persistence gate**: a binary head predicting, for each of the 64 positions, whether that token changes at all on this step, decoded *before* token identity. Positions the gate marks unchanged are copied from the previous frame verbatim and never sampled, so they cannot flicker; only the remainder go through the usual decode. This attacks the exponent rather than the accuracy, because stillness stops requiring 64 simultaneously-correct predictions and starts requiring one well-calibrated binary decision per token, on a task where the prior is heavily skewed and easy to learn. It also cuts decode work on quiet frames. Not implemented, and not on the critical path for the scaling ladder.

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
