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

## Amendments of 2026-08-19, added before any scaled run

**Seeds and resolution.** The 2M rung runs three times with seeds 0, 1, 2 at
identical settings. The spread (max minus min) of one-step PSNR on moving
transitions across those seeds is the **resolution of the ladder**: any
rung-to-rung difference smaller than that spread is read as "no effect", and
every rule above is read against it. The same spread qualifies the other
reported metrics wherever a rule compares rungs.

**Learning rate across rungs, decided before results.** A single shared LR
would let the 26M rung underperform for optimization reasons and fire rule 1
"structural" on a hyperparameter artifact. Policy: the 2M rung keeps its tuned
6.0e-4 (d_model 192). At 8M, a three-point LR probe runs first -- the
width-scaled anchor `6.0e-4 * sqrt(192/384) = 4.2e-4`, twice it, and half it --
each for ~0.4 epochs, winner by held-out loss at matched tokens. The full 8M
run uses the winner, and 26M uses the winner scaled by `sqrt(384/512)`. If the
probe winner is an endpoint of the three, the probe extends one more point in
that direction before 26M starts.

**Context axis, interpretation bound.** The context rungs run at 8M because
that is the affordable capacity, and the bound is stated now: a null context
result at 8M supports only "8M cannot exploit longer context on this task",
not "context does not matter". Upgrading to the general claim would require
re-running the context axis at the best capacity, which is a separate,
costed decision.

## Amendment of 2026-09-10, after 2M seed 0 and before any other run

**The plateau rule did not fire, and the ladder re-bases on a fixed token
budget.** 2M seed 0 ran a 16-epoch cosine (the 58k-step cap had been sized for
batch 256 and became 16 epochs at the batch actually used, 512). Held-out loss
improved by 0.005 to 0.02 per 1000 steps for the entire schedule and ended at
1.920 after 15.6 epochs, so "to convergence" collapsed into "to the cap", and
matched tokens at that budget would put the 26M rung far outside the $60 cap.

Every rung from here, including a fresh 2M, trains at **T\* = 4 epochs of the
2M-frame dataset = 2.916B tokens** (tokens = windows x context x 65), with the
cosine schedule planned over exactly that budget so it decays to 5% at the end.
The three 2M seeds at T\* define the ladder's resolution as before. All T\*
numbers are "at 4 epochs" by construction; a rung that would have kept
improving is equally under-trained relative to every other rung, which is the
comparison the ladder is for.

Seed 0's 16-epoch run is kept as the **long-schedule reference**. It already
answers rule 1's starved-vs-structural question for 2M: 64% of headroom at 4x
the T\* budget against 20% for the CPU checkpoint. The T\* ladder is therefore
mainly answering rule 2, which axis moves which metric at equal tokens.

Batch sizes per rung are chosen for memory (512 at 2M, smaller above) and the
budget fixes the step count, so batch is not a confound in tokens seen. The 8M
LR probe stands as written.

## Amendment of 2026-09-13, before any 8M or 26M result is read

Measurement fixes only. The rules, the rungs and the token budget above are
unchanged. The 8M learning-rate probe was already running when this was
written; its losses are used only to pick the learning rate, as the probe
policy says, and nothing below depends on them.

**Every rung is scored on its final checkpoint.** The report used to load
`dynamics.pt`, the best-by-validation checkpoint, which the trainer picks on a
20-batch validation slice. For `ladder-2m-t4-s2` that was step 14000 at 3.83
epochs, so one of the three seeds that set the resolution was not at T\*.
Scoring `final.pt` puts every rung at exactly T\*, as the 2026-09-10
amendment intends.

**Held-out loss comes from fixed frames.** The trainer's validation loss reads
the first 20 batches of the unshuffled validation split, so how much of it a
rung sees depends on its batch size: 10,240 windows at batch 512, 1,280 at
batch 64. `ngx.eval.heldout` scores 4,096 target frames spread over the whole
split, the same frames for every rung, with seeded masks. The held-out loss
and cold accuracy columns come from it, and the 8M probe is decided on it.

**Context rungs predict the same frames.** One-step windows, closed-loop
windows and revisit pairs were drawn separately for each context length, so a
24-frame rung would have been scored on mostly different target frames than a
6-frame rung. Windows are now drawn once with 24 frames of history and each
rung is handed its own last C frames; revisit pairs are found from frame 24
and every drift rollout starts there.

All existing rows are rescored under this protocol, including the three 2M
seeds, so the resolution is measured again before any rung is compared with it.

## Clarifications of 2026-09-14, after the ladder ran

Wording only. No rule, rung, budget or reading changes.

- The 29.53 dB ceiling quoted under *The ladder* has no committed source.
  Headroom is computed against the tokenizer round trip on the same moving
  windows as the model, which under the 2026-09-13 protocol is 30.97 dB (see
  [LADDER.md](LADDER.md)).
- "All T\* numbers are at 4 epochs by construction" holds at a 6-frame
  context. The budget is matched in tokens, so longer-context rungs see
  proportionally fewer windows: 2.03 epochs at 12 frames.
- The capacity rung written as 30M above is the 26M config (d_model 512,
  8 layers, 25.7M parameters).

## What is not pre-registered

Anything not written above. If a rung produces something surprising outside
these rules, it is reported as an observation, not as a confirmation of
anything, and it does not retroactively become a hypothesis.
