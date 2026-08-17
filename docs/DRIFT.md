# Drift

Reference trajectory: 1064 real frames from one unbroken episode, explorer policy, env seed 1. Every number below is the mean over 4 rollouts with different sampling seeds, +/- one standard deviation. Decoding samples, so a single rollout cannot tell an effect from noise -- and on this checkpoint, the difference between the two configurations is noise.

## Divergence from the real game

PSNR between the model's frame and the game's frame at step *k*, both driven by the same actions from the same starting frames.

| config | k=1 | k=10 | k=25 | k=50 | k=100 | k=250 | k=500 | k=1000 |
|---|---|---|---|---|---|---|---|---|
| sliding context only | 16.2 | 14.2 | 13.1 | 10.6 | 10.6 | 13.2 | 9.1 | 7.9 |
| memory | 16.2 | 14.2 | 13.1 | 10.6 | 10.6 | 13.2 | 8.7 | 8.6 |

## Return-to-place consistency

Pairs of steps where the real player stood within 40 map units and 20 degrees of a pose from at least 60 steps earlier (median actual gap: 507 frames -- far outside the model's 6-frame context, so nothing but memory can carry the room across). `game` is the same measurement on the real frames: the ceiling, since matching poses are close but never identical.

| config | pairs | model | game (ceiling) | gap | retrieval fired |
|---|---|---|---|---|---|
| sliding context only | 40 | **9.89 +/- 0.54 dB** | 11.25 dB | 1.36 dB | 0/1000 frames |
| memory | 40 | **9.20 +/- 0.88 dB** | 11.25 dB | 2.05 dB | 497/1000 frames |

## What this says

**Retrieval memory does not help at this scale.** It moves the return-to-place score by -0.69 dB against a run-to-run spread of +/-1.03 dB, which is to say it does not move it. Reporting the single best seed would have shown an improvement; four seeds show that improvement was the seed.

Two things are worth separating here, because only one of them is a dead end.

*Retrieval itself works.* On the reference trajectory the correct past frame is the top-ranked match for 55% of genuine revisits and in the top two for ~80%. The mechanism finds the room.

*The model cannot use it.* At 2.0M parameters and under one pass over the data, predictions are dominated by the most recent frame; replacing a distant context slot with a remembered one perturbs an input the model is barely conditioning on. The ~45% of retrievals that surface the wrong room then cost roughly what the right ones gain, which is exactly the wash the table shows.

So `memory.enabled` ships as `false`. The feature stays in the codebase and on the `M` key, because the premise it is built on -- that an 8-frame context cannot hold a room you left 500 frames ago -- is unchanged, and a model with enough capacity to exploit distant context is the obvious thing to re-test it against. Claiming it as a win on this checkpoint would just be reporting noise.

One structural note on the curve above: `exclude_recent` blocks retrieval until 64 writes have accumulated, so the two configurations are identical by construction for the first few hundred frames. The early columns matching exactly is expected, not a bug.

Regenerate with `python -m ngx.eval.drift --config configs/small.yaml`.
