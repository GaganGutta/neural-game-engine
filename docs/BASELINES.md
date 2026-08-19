# B0: baselines

Model: 2.0M params, context 6 frames, val loss 3.203, cold token accuracy 0.152. Greedy decoding, 4-pass MaskGIT, memory off.

## One-step, teacher-forced

Real frames for context, one frame predicted, 400 held-out windows. This is the test that cannot be excused by drift.

Split by whether the two real frames are pixel-identical. On those, copy-last-frame is exactly right and PSNR is infinite, so any aggregate that mixes them is decided by where the infinity is clipped rather than by the models. Splitting removes the clipping choice from the comparison instead of managing it.

### Moving transitions (387 windows, 96.8%)

No infinities here, so no cap exists and the mean is well defined.

| predictor | mean PSNR | median PSNR | token accuracy |
|---|---|---|---|
| copy-last-frame | 20.96 dB | 21.24 dB | 0.097 |
| model (2.0M), greedy | **23.01 dB** | 23.50 dB | 0.152 |
| model (2.0M), sampled | 22.94 dB | 23.69 dB | 0.120 |
| tokenizer ceiling | 31.03 dB | 30.84 dB | 1.000 |

**The model beats copy-last-frame by 2.05 dB on moving transitions.** Headroom from copy to the tokenizer ceiling is 10.07 dB, so it captures **20% of what was available**. Token accuracy agrees and is cap-free by construction.

### Static transitions (n = 13 windows, 3.2% of the sample)

Frames where nothing moved: a no-op action, or the agent pressed against a wall.

| predictor | mean PSNR | median PSNR | token accuracy |
|---|---|---|---|
| copy-last-frame | exact (infinite) | exact | 1.000 |
| model (2.0M), greedy | 30.33 dB | 29.31 dB | 0.728 |
| tokenizer ceiling | 31.35 dB | | 1.000 |

**The model does not beat copy-last-frame here, and cannot.** Copy is exact by construction; the model scores 30.33 dB. It reproduces the previous frame's tokens exactly on **0%** of static transitions (n = 13).

Read those two numbers differently. At n = 13 the 0% exact-repeat result is decisive: not one static frame in 13 came back unchanged. The 30.33 dB mean is not; it is a small sample and should be treated as indicative only.

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

### Churn under a held no-op, against the real game

The static row above is teacher-forced. This is the same question asked the way a viewer meets it: nothing pressed for 60 frames, the model consuming its own output. The real game is run through the same prefix, the same no-op and the same tokenizer from the same starts, because a churn number on its own supports either reading, too twitchy or too sticky. Start(s) [3] ended the episode on the first no-op frame (residual momentum carried the player onto the goal) and are excluded from both rows.

**What the real game's changes under a no-op actually are.** Every one of them was identified before the numbers were trusted. They are *settling*: residual momentum sliding to a stop (position deltas decaying 6.8, 4.6, 3.1, 2.1, 1.4, ... map units per frame after a forward action) and the view height returning to rest (a 1 to 3 row vertical shift of the whole frame with zero horizontal motion). All of it is over within about ten frames. No hold-still window crossed an episode boundary. The map does contain a small animated object, 37 pixels in a 9x8 region alternating between two states every 2 to 3 frames, visible from one of the cold-start spawns; none of the matched starts used here had it in view after the prefix, so it does not enter this table, but a model that learned it would show tiny periodic churn that looks like twitchiness unless you know where the player is standing.

Because the events are settling, the pairs are bucketed by how many no-op frames have been held. An aggregate over 60 frames hides everything that matters here.

| held for | pairs | real: identical | real: tokens changed when it moves | model: identical | model: tokens changed when it moves |
|---|---|---|---|---|---|
| k=1 | 7 | 14.3% (6 events) | 36.2 / 64 | 0.0% (7 events) | 26.1 / 64 |
| k=2-10 | 63 | 79.4% (13 events) | 32.7 / 64 | 50.8% (31 events) | 4.3 / 64 |
| k>10 | 350 | 100.0% (0 events) | 0.0 / 64 | 99.7% (1 events) | 1.0 / 64 |
| all | 420 | 95.5% (19 events) | 33.8 / 64 | 90.7% (39 events) | 8.1 / 64 |

**Steady state (k>10, 350 pairs): the model is still.** 99.7% identical against the real game's 100.0%, 1 change events in 350 pairs. Whatever the model's problem is, it is not background shimmer, and an earlier version of this section said it was.

**Settling window (k=1 to 10): the model changes at the right times and by too little.** Its change events fall in the same first-ten-frame window as the real game's, and nowhere else. On starts where the player was sliding, it fires on roughly the same frames the game does but moves ~4 of 64 tokens where the game moves ~33. On quiet starts, where the game shows only the k=1 view-height settle, the model adds a few 1-to-3-token twitches over the first several frames. So the model is too twitchy in *count* inside the settling window (31 events vs 13 in k=2-10) and far too timid in *magnitude*. That is a sharpness failure on the momentum and view-bob dynamics, confined to the moments right after motion stops.

**This is much better than the teacher-forced static row predicts, and the discrepancy is the interesting part.** Teacher-forced, the model reproduces a real previous frame's tokens exactly 0% of the time. Closed-loop and past the settling window, it reproduces its own previous tokens 99.7% of the time. Those are different tasks: matching an external frame exactly is hard, while settling on a self-consistent fixed point is something an autoregressive model falls into, because its context is already full of the frame it just produced. Self-consistency rather than accuracy is doing the work, and neither is what the loss optimises, so this can move either way with a stronger model and is re-measured on every rung.

These are the before-numbers for any later fix.

### Candidate fix, logged and not implemented

A per-token **persistence gate**: a binary head predicting, for each of the 64 positions, whether that token changes at all on this step, decoded *before* token identity. Positions the gate marks unchanged are copied from the previous frame verbatim and never sampled, so they cannot flicker; only the remainder go through the usual decode. This attacks the exponent rather than the accuracy, because stillness stops requiring 64 simultaneously-correct predictions and starts requiring one well-calibrated binary decision per token, on a task where the prior is heavily skewed and easy to learn. It also cuts decode work on quiet frames. Not implemented, and not on the critical path for the scaling ladder.

## Closed-loop

8 independent rollouts of 32 frames from held-out starts, averaged per step. The model consumes its own predictions; copy-last-frame degenerates to freezing on the last real frame. Reported per k and never averaged over k, because the trajectory is not monotonic and a single mean over it would describe nothing. PSNR is clipped at 100 dB in this section only, for the rare case of a rollout that starts from a stalled agent.

| k | copy (frozen) | model | tokenizer ceiling | model lead |
|---|---|---|---|---|
| 1 | 22.04 dB | **25.00 dB** | 30.87 dB | +2.96 dB |
| 2 | 21.25 dB | **23.59 dB** | 30.40 dB | +2.34 dB |
| 4 | 22.91 dB | **24.21 dB** | 30.58 dB | +1.29 dB |
| 8 | 19.36 dB | **22.46 dB** | 30.92 dB | +3.10 dB |
| 16 | 17.77 dB | **20.31 dB** | 30.57 dB | +2.54 dB |
| 32 | 13.16 dB | **15.41 dB** | 31.12 dB | +2.24 dB |

The model's lead over a frozen frame decays from +2.96 dB at k=1 to +2.24 dB at k=32. Past roughly k=16 it is not meaningfully better than showing the player a still image, which is the honest way to read the rollout GIF.

The tokenizer ceiling is flat across k, as it must be: it re-encodes the true frame at every step and never compounds.

Regenerate with `python -m ngx.eval.baselines --config configs/small.yaml`.
