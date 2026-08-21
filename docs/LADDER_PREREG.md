# Pre-registration for the scaling ladder

Written and committed on 2026-08-19, before any scaled model exists. The point
of writing it now is that the readings below cannot be adjusted to fit the
numbers once the numbers arrive. Every rung reports against every rule here,
and each rule says which branch fired.

Baseline the ladder is measured against: the 2.0M-parameter, 6-frame-context
model in `checkpoints/small/`, trained on 150k frames for under one epoch, whose
numbers are in [BASELINES.md](BASELINES.md), [ACTION_ABLATION.md](ACTION_ABLATION.md)
and [DRIFT.md](DRIFT.md).

## The ladder

Two axes, run as a controlled experiment.

**Capacity, at fixed 6-frame context.** 2M trained to convergence (held-out
loss plateau, not a step count), then 8M and 30M at *matched tokens seen*.
Matched tokens, not matched wall clock, or capacity and data seen are
confounded.

**Context, at the best capacity from above.** 6, 12 and 24 frames at matched
tokens seen.

Every config reports: params, context frames, tokens seen, epochs, held-out
loss, one-step PSNR on moving transitions, percent of headroom captured against
the 29.53 dB tokenizer ceiling (moving subset, no cap involved), closed-loop
lead over a frozen frame at k=1, 8, 16, 32, return-to-place consistency, and
the bucketed hold-still table.

## Rule 1: starved or structural

Fires on the capacity axis.

- **Sound and starved:** percent-headroom-captured rises across 2M to 8M to
  30M, and the 2M-to-convergence run alone lands well above the current 20%.
  Continue scaling.
- **Structural:** the 2M model converges and still captures under roughly 25%
  of headroom, or the ladder flattens between 8M and 30M. Stop spending. Debug
  the objective, the context length, or the action conditioning instead.

## Rule 2: which axis moves which metric

Stated before any number is seen.

- One-step PSNR should respond to **capacity** and be roughly flat in
  **context**. It is a local prediction and six frames is enough history for
  it.
- Closed-loop lead at k=16 and k=32, and return-to-place consistency, should
  respond to **context** and be roughly flat in **capacity**. A 6-frame window
  cannot know it has returned to a room it saw 40 frames ago no matter how many
  parameters it has.

**If both metrics move with capacity and neither moves with context, the
diagnosis behind the context axis is wrong**, and the report says so plainly
rather than fitting a story to it. The candidate wrong-diagnosis story is that
retrieval memory came back null because there was no coherent model to
retrieve into, not because the window was too short; that would show up as
return-to-place responding to capacity.

## Rule 3: stillness, read bucketed and read together

The hold-still comparison is reported per rung, bucketed by how long the no-op
has been held, against the real-game reference:

| held for | real: identical | real: tokens changed when it moves |
|---|---|---|
| k=1 | 14.3% | 36.2 / 64 |
| k=2-10 | 79.4% | 32.7 / 64 |
| k>10 | 100.0% | 0 events |

The current model at k=2-10 is 50.8% identical with 4.3 tokens when it moves:
too many events, each far too small.

A stronger model should move **both** numbers in the k=2-10 bucket toward the
reference **together**: identical rate up toward 79% *and* tokens-changed-when-
it-moves up toward 33. That is what modelling the momentum and view-bob decay
correctly looks like: fewer, larger, correctly timed changes.

- **Sharper:** identical rate up and churn-when-moves up. The model learned the
  settling dynamics.
- **Stickier:** identical rate up while churn-when-moves stays near 4. The
  model is suppressing changes rather than learning them, and the demo will
  read as sluggish response to input rather than as shimmer.
- **Twitchier:** identical rate down. Worse.

Each rung names which of the three fired. The k>10 bucket is expected to stay
at or near 100% on every rung; a drop there is a regression and is reported as
one.

## Rule 4: eviction and cache carrying

*Amended 2026-08-19, before any scaled run:* the cadence sweep was dropped by
decision. Never-refreshed carrying diverges materially from full recompute
(40% of frames identical over 60-frame rollouts, untrained model), so the
default is a full cache rebuild at every frame boundary, which is exact.
Carrying stays as one benchmark row per rope rung; it may be re-read on a
trained checkpoint but no sweep machinery exists and none is planned.

## What is not pre-registered

Anything not written above. If a rung produces something surprising outside
these rules, it is reported as an observation, not as a confirmation of
anything, and it does not retroactively become a hypothesis.
