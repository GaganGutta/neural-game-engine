# Notes on building a playable world model

The goal was narrow: press W and have a network draw the next frame, fast
enough that it feels like a game rather than a slideshow. Everything below is a
consequence of that.

## The sequence layout is the whole design

A frame becomes 64 tokens through a VQ-VAE. A window of frames plus the actions
between them has to become one sequence. The obvious layout is

```
[f_0][a_0][f_1][a_1] ... 
```

with a causal mask, and it works, but it forces you into one token at a time at
inference: 64 sequential forward passes per frame. At 64x64 that is the
difference between a demo and a stutter.

MaskGIT fixes the pass count by predicting many tokens at once, which needs
*bidirectional* attention inside the frame being predicted. That collides with
the causal mask: if tokens inside frame `t` can see each other, then frame
`t+1` can see frame `t`'s tokens through them, and a partially-masked frame `t`
means later frames attend to a corrupted history that never occurs at play
time.

The fix here is two streams in one sequence.

```
stream A (clean context)   [ f_0 ... a_0 ][ f_1 ... a_1 ] ...   block = L+1
stream B (masked targets)  [ ~f_1 ][ ~f_2 ] ...                 block = L
```

* A block `t` attends to A blocks `0..t`, bidirectionally inside its own block.
* B block `s` attends to A blocks `0..s-1` and to itself. Nothing else.

Context is always clean, because the masking happens in a *copy*. And because
every B block is independent, one forward pass supervises every frame in the
window rather than just the last one. The cost is roughly 2x sequence length
for `T-1` times the supervision, which is a good trade at these sizes.

The action sits at the *end* of its block, not the start. That is what makes
"everything before block `t+1`" equal "everything a player knows when frame
`t+1` is about to be drawn". The layout encodes the causality instead of
relying on a comment to assert it.

## Two decoders, one set of weights

Because B blocks are bidirectional over a partially-masked frame, the trained
model supports two decoding strategies with no retraining and no second
objective:

* **raster**: reveal one token per pass, left to right. 64 passes. This is the
  autoregressive baseline.
* **MaskGIT**: every pass predicts all remaining slots, keep the most
  confident ones, repeat. 4 passes in the shipped configs.

They are the same weights on the same mask, which is what makes the speed
comparison in [BENCHMARKS.md](BENCHMARKS.md) fair. If the fast path needed its
own training run, the comparison would be measuring two models, not two
decoders.

## What the KV cache actually caches

During the decoding of one frame, stream A does not change. Only the target
block does. So stream A's keys and values are computed once per frame and
reused across all 4 (or 64) passes. Each pass then runs 64 query positions
against a cached 390-position prefix instead of recomputing that prefix on
every pass.

This is exact, not approximate, and the benchmark proves it rather than
asserting it: the cached and uncached rows produce bit-identical rollouts under
greedy decoding, which shows up in the table as `identical` (an infinite PSNR).

`tests/test_dynamics.py` pins the same property at the logit level: the
training forward pass and the cached inference path must agree to 1e-4. Without
that test, a subtly wrong cache would still produce plausible-looking Doom, and
plausible-looking Doom is exactly what a broken world model looks like.

## Drift, and why memory is placed where it is

A short sliding context (6 frames in the shipped config, 16 in `full.yaml`)
means everything the model knew about a room is gone moments after you leave.
Walk out, walk back, and the room is regenerated from nothing, usually as a
*different* room. The rollout stays plausible while ceasing to be consistent.
That is the characteristic failure, and it is not fixed by training longer.

The countermeasure is a retrieval memory keyed on a frame's mean codebook
embedding: the average of the embedding vectors of its 64 tokens,
L2-normalised, compared by cosine similarity. No extra network, no extra
training.

The placement matters more than the mechanism. Retrieved frames **replace the
oldest context slots** rather than extending the context. The block count stays
exactly what the model was trained on, so the retrieval layer needs no
architectural change and no fine-tuning. It is a pure inference-time addition
that can be toggled with a keypress while playing.

Two guards keep it honest:

* it ignores anything written in the last `exclude_recent` writes, so the
  nearest neighbour is never the frame from two steps ago;
* it declines below a similarity threshold rather than returning the least-bad
  match, because injecting a wrong room is worse than injecting nothing.

## Measuring drift without fooling yourself

Two numbers, because there are two different failure modes.

**Divergence from the real game.** Run the model and the real game under
identical actions from identical seed frames, and track PSNR. This decays no
matter what; the model samples, it does not simulate. The shape of the curve is
the signal, not the floor.

**Return-to-place consistency.** Find two moments where the real player stood
in the same spot facing the same way, separated by a long gap, and ask whether
the model drew the same room both times. This is the number retrieval memory is
built to move.

The second metric needs a control, and getting this wrong is the easy mistake.
"Same pose" is a tolerance, not an identity, so even the *real game* does not
score infinite PSNR between two matched moments. The evaluation therefore
reports the real game's own return-to-place PSNR alongside the model's. The
model's number is only interpretable relative to that ceiling.

It also needs the two visits to be genuinely far apart. On the reference
trajectory the median gap between matched poses is several hundred frames,
orders of magnitude beyond the context window, so nothing except memory can
carry the room across.

One more trap: drift has to be measured inside a *single* episode. `my_way_home`
ends the moment the player stumbles onto the goal, which for an explorer policy
can happen after 200 steps or not for 1500. Splicing two episodes together
would put a teleport in the middle of the trajectory and score it as drift, so
the evaluation searches seeds for one long enough episode and records which seed
it used.

## What the benchmark harness refuses to do

* No cherry-picked warmup. Warmup frames are discarded explicitly, which
  matters enormously for `torch.compile`.
* No sampling during timing. Greedy decoding, so every row produces a
  comparable rollout instead of a different one.
* No silent quality loss. Every row reports PSNR against the reference rollout,
  so a change that buys throughput by degrading output is visible in the same
  table as the throughput.
* No silent failure. A row that cannot run on this machine is printed as
  unavailable with the reason attached.
* No dragging regressions forward. A change that measures slower is reverted
  and labelled, and later rows build on the best configuration. Not every
  optimisation survives contact with a given machine, and the table says which
  ones did not.

## What the measurements said, including where I was wrong

Three things did not go the way the design assumed. All three are in the docs
with numbers attached rather than quietly fixed.

**Retrieval memory does not help, and the retrieval itself is weak.** At 2M it is a
wash: -0.69 dB on return-to-place against a +/-1.03 dB run-to-run spread. The
first version of this evaluation ran one rollout per configuration and showed
memory winning; four seeds showed that the win was the seed. I first blamed the
model, arguing that a 2.0M-parameter model leans too hard on the most recent
frame to use a distant context slot. The 26M checkpoint cannot settle that
either way: its sliding context alone already scores 11.24 dB against the
game's own 11.25 dB, so there is no gap for memory to close, and memory is
again not measurable (-0.64 dB against +/-1.69 dB). Measuring the key directly
points at a problem that holds at both scales. At a genuine revisit, the most similar stored frame was
taken at the same place only 11% of the time, and one of the top two 17% of
the time, so at most revisits neither of the two best matches in memory was taken at
that place. See
[DRIFT.md](DRIFT.md) and [DRIFT_26M.md](DRIFT_26M.md).

**The retrieval key had to change, and it is still the weak link.** The
original bag-of-codes histogram scored matched revisits too low for a
similarity threshold to mean anything. Mean-pooled codebook embeddings fixed
the *scale* by using the metric structure the codebook already learned. They
did not fix the *ranking*: at a revisit the most similar stored frame is
usually from somewhere else. A geometry-aware key is the next thing to try.

**More decoding passes buy nothing here.** The premise of stage 4 was that
MaskGIT trades quality for speed and the job is finding the sweet spot. On this
checkpoint there is no trade to make: PSNR drifts slightly *down* from 1 pass to
64, and sharpness is flat at ~0.52x the real frame's detail across every pass
count. The blur is not coming from the decoder, so no decoding schedule can fix
it. I initially wrote that sharpness would rise with pass count; it does not,
and [DECODE.md](DECODE.md) says so. `maskgit_steps` is 4 because extra
passes buy nothing measurable on this checkpoint, so more of them would only
cost latency. That sweep is from the 2M checkpoint and has not been re-run at
26M.

What did work as designed: the KV cache is exactly what it claims to be
(bit-identical rollouts, 6.3x), and MaskGIT over raster is an 8.6x win on top
of that: 0.65 fps to 35.47 fps end to end on the laptop CPU. At 26M the same
table goes from 0.09 to 10.87 fps, and there bf16 and int8 pay for themselves
where at 2M they did not. See [BENCHMARKS.md](BENCHMARKS.md) and
[BENCHMARKS_26M.md](BENCHMARKS_26M.md).

## Honest limitations

* The shipped checkpoint was trained on a laptop CPU. It is far below the scale
  in `configs/full.yaml`, and it looks like it: expect a recognisable but soft
  world that drifts off the real game within about 50 frames. The pipeline, not the
  checkpoint, is the artifact.
* int8 dynamic quantisation is CPU-only here. The GPU equivalent is a different
  toolchain (torchao, bitsandbytes, TensorRT) and is not implemented.
* Retrieval is keyed on appearance, not geometry. Two corridors with the same
  texture and lighting are, to this memory, the same place. A learned or
  pose-aware key would separate them; a mean of codebook embeddings does not,
  which is most of why retrieval picks the wrong place.
* Splicing a remembered frame into slot 0 makes that slot's transition a
  fiction: the action stored with it did not produce the frame now sitting in
  slot 1. The retrieved frames are there to put the room's textures back in
  view, and the junction is an accepted approximation rather than a modelled
  one. Whether the trade pays is an empirical question, which is why
  [DRIFT.md](DRIFT.md) reports memory on and memory off side by side instead of
  assuming the answer.
* One scenario is wired end to end. Other VizDoom scenarios are supported by
  the env wrapper but only `my_way_home` has been trained and evaluated.
