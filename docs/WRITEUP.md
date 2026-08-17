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
`t+1` is about to be drawn" — the layout encodes the causality instead of
relying on a comment to assert it.

## Two decoders, one set of weights

Because B blocks are bidirectional over a partially-masked frame, the trained
model supports two decoding strategies with no retraining and no second
objective:

* **raster** — reveal one token per pass, left to right. 64 passes. This is the
  autoregressive baseline.
* **MaskGIT** — every pass predicts all remaining slots, keep the most
  confident ones, repeat. 8 passes.

They are the same weights on the same mask, which is what makes the speed
comparison in [BENCHMARKS.md](BENCHMARKS.md) fair. If the fast path needed its
own training run, the comparison would be measuring two models, not two
decoders.

## What the KV cache actually caches

During the decoding of one frame, stream A does not change. Only the target
block does. So stream A's keys and values are computed once per frame and
reused across all 8 (or 64) passes. Each pass then runs 64 query positions
against a cached prefix instead of ~1000 positions from scratch.

This is exact, not approximate, and the benchmark proves it rather than
asserting it: the cached and uncached rows produce bit-identical rollouts under
greedy decoding, which shows up in the table as an infinite PSNR.

`tests/test_dynamics.py` pins the same property at the logit level — the
training forward pass and the cached inference path must agree to 1e-4. Without
that test, a subtly wrong cache would still produce plausible-looking Doom, and
plausible-looking Doom is exactly what a broken world model looks like.

## Drift, and why memory is placed where it is

An 8-frame context means everything the model knew about a room is gone eight
frames after you leave. Walk out, walk back, and the room is regenerated from
nothing — usually as a *different* room. The rollout stays plausible while
ceasing to be consistent. That is the characteristic failure, and it is not
fixed by training longer.

The countermeasure is a retrieval memory keyed on the bag-of-codes histogram of
a frame: which codebook entries appear in it, L2-normalised, compared by cosine
similarity. It works because the tokenizer already spends different codes on
different wall textures, which is precisely what distinguishes one room in this
maze from another. No extra network, no extra training.

The placement matters more than the mechanism. Retrieved frames **replace the
oldest context slots** rather than extending the context. The block count stays
exactly what the model was trained on, so the retrieval layer needs no
architectural change and no fine-tuning — it is a pure inference-time addition
that can be toggled with a keypress while playing.

Two guards keep it honest:

* it ignores anything written in the last `exclude_recent` writes, so the
  nearest neighbour is never the frame from two steps ago;
* it declines below a similarity threshold rather than returning the least-bad
  match, because injecting a wrong room is worse than injecting nothing.

## Measuring drift without fooling yourself

Two numbers, because there are two different failure modes.

**Divergence from the real game** — run the model and the real game under
identical actions from identical seed frames, and track PSNR. This decays no
matter what; the model samples, it does not simulate. The shape of the curve is
the signal, not the floor.

**Return-to-place consistency** — find two moments where the real player stood
in the same spot facing the same way, separated by a long gap, and ask whether
the model drew the same room both times. This is the number retrieval memory is
built to move.

The second metric needs a control, and getting this wrong is the easy mistake.
"Same pose" is a tolerance, not an identity, so even the *real game* does not
score infinite PSNR between two matched moments. The evaluation therefore
reports the real game's own return-to-place PSNR alongside the model's. The
model's number is only interpretable relative to that ceiling.

It also needs the two visits to be genuinely far apart. On the reference
trajectory the median gap between matched poses is several hundred frames —
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

## Honest limitations

* The shipped checkpoint was trained on a laptop CPU. It is far below the scale
  in `configs/full.yaml`, and it looks like it: expect a recognisable but soft
  world that drifts within a few hundred frames. The pipeline, not the
  checkpoint, is the artifact.
* int8 dynamic quantisation is CPU-only here. The GPU equivalent is a different
  toolchain (torchao, bitsandbytes, TensorRT) and is not implemented.
* Retrieval is keyed on appearance, not geometry. Two corridors with the same
  texture and lighting are, to this memory, the same place. A learned or
  pose-aware key would separate them; a bag-of-codes histogram will not.
* Splicing a remembered frame into slot 0 makes that slot's transition a
  fiction: the action stored with it did not produce the frame now sitting in
  slot 1. The retrieved frames are there to put the room's textures back in
  view, and the junction is an accepted approximation rather than a modelled
  one. Whether the trade pays is an empirical question, which is why
  [DRIFT.md](DRIFT.md) reports memory on and memory off side by side instead of
  assuming the answer.
* One scenario is wired end to end. Other VizDoom scenarios are supported by
  the env wrapper but only `my_way_home` has been trained and evaluated.
