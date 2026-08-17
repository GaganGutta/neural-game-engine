# neural-game-engine

**A game you play inside a neural network.** Train an action-conditioned world
model on Doom, then throw the game engine away and drive the model with your
keyboard. Every pixel below the right-hand label was predicted by a transformer
from the last few frames plus a keypress.

<!-- DEMO_GIF -->

```bash
python play.py
```

## What is actually running

At play time there is no Doom. There is a rolling window of the last few
frames, an action index from the keyboard, and a transformer that predicts the
next frame's tokens. The real game is used for exactly one thing: supplying the
handful of frames that prime the context when you start (and when you press R).

```
   keyboard ─┐
             ▼
  ┌────────────────────────────────────────────────────────┐
  │  last C frames ──► VQ-VAE encoder ──► 64 tokens/frame  │
  │                                            │           │
  │                              dynamics transformer      │
  │                                            │           │
  │                              next frame's 64 tokens    │
  │                                            │           │
  │                          VQ-VAE decoder ──► 64x64 RGB  │
  └────────────────────────────────────────────────────────┘
             ▲                                   │
             └───────────  fed back  ────────────┘
```

The feedback loop is the whole problem. Each predicted frame becomes context
for the next one, so errors compound: the model does not simulate the world, it
samples a plausible continuation of it, and plausible continuations drift.
Sections below measure how fast it drifts and what claws it back.

## Quickstart

```bash
pip install -r requirements.txt
```

Then either play the shipped checkpoint:

```bash
python play.py
```

or build everything from scratch (see [reproducing](#reproducing) for timings):

```bash
bash scripts/reproduce.sh configs/small.yaml
```

Controls: `W`/`↑` walk, `A`/`D` turn, hold `W` with a turn to round a corner.
`R` reseeds from the real game, `M` toggles retrieval memory, `TAB` switches
between MaskGIT and raster decoding, `ESC` quits.

## The pipeline

| stage | what it does | entry point |
|---|---|---|
| 1. data | scripted + random policies through VizDoom, `(frame, action)` pairs at 64x64 | `ngx.data.collect` |
| 2. tokenizer | VQ-VAE, 64x64 RGB down to an 8x8 grid of codebook indices | `ngx.train.train_vqvae` |
| 3. dynamics | causal transformer over frame tokens + action embeddings | `ngx.train.train_dynamics` |
| 4. speed | KV cache, MaskGIT parallel decode, bf16, int8 | `ngx.eval.bench` |
| 5. drift | sliding context + retrieval memory, measured | `ngx.eval.drift` |

The one design decision worth reading about is the sequence layout — two
streams in one sequence, so context stays clean while the target frame is
masked, which is what lets a single set of weights serve both a 64-pass
autoregressive decoder and an 8-pass parallel one. That is in
[docs/WRITEUP.md](docs/WRITEUP.md).

## Making it fast

<!-- BENCH_TABLE -->

## Fighting drift

An 8-frame context means everything the model knew about a room is gone eight
frames after you leave it. Walk out, walk back, and the room gets regenerated
from nothing — usually as a *different* room. The rollout stays plausible while
ceasing to be consistent.

The countermeasure is a retrieval memory keyed on the bag-of-codes histogram of
each frame, which works because the tokenizer already spends different codes on
different wall textures. Retrieved frames **replace the oldest context slots**
rather than extending the context, so the model sees exactly the block layout
it was trained on and needs no retraining — you can toggle it mid-game with `M`.

<!-- DRIFT_TABLE -->

## Reproducing

<!-- REPRO_TIMINGS -->

## Repo layout

```
ngx/
  envs/          VizDoom wrappers, discrete action sets, pose (eval only)
  data/          collection, tokenization, datasets
  models/        vqvae.py, dynamics.py
  train/         train_vqvae.py, train_dynamics.py
  infer/         engine.py (KV cache, decoders), memory.py, quantize.py
  eval/          bench.py, drift.py
configs/         small.yaml (CPU), full.yaml (one GPU)
tests/           train/inference equivalence, cache exactness, memory behaviour
play.py          the demo
```

## Tests

```bash
python -m pytest tests/ -q
```

The one that matters is `test_cached_inference_matches_training_forward`:
`play.py` never runs the training forward pass, so if the cached path and the
training path ever disagree, the model you play is not the model you trained —
and the failure is silent, because a subtly wrong world model still produces
plausible-looking Doom.

## Limitations

<!-- LIMITATIONS -->

## License

MIT, see [LICENSE](LICENSE).
