# Drift

Checkpoint: `runs/ladder-26m-t4-s0/dynamics/final.pt`, 25.7M parameters, 6-frame context, trained on 2.92B tokens (4.00 epochs).

Reference trajectory: 1064 real frames from one unbroken episode, explorer policy, env seed 1. Every number below is the mean over 4 rollouts with different sampling seeds, +/- one standard deviation. Decoding samples, so a single rollout cannot tell an effect from noise.

## Divergence from the real game

PSNR between the model's frame and the game's frame at step *k*, both driven by the same actions from the same starting frames.

| config | k=1 | k=10 | k=25 | k=50 | k=100 | k=250 | k=500 | k=1000 |
|---|---|---|---|---|---|---|---|---|
| sliding context only | 30.5 | 20.6 | 13.8 | 11.3 | 10.5 | 12.7 | 10.6 | 11.7 |
| memory | 30.5 | 20.6 | 13.8 | 11.3 | 10.5 | 12.7 | 11.2 | 9.2 |

## Return-to-place consistency

Pairs of steps where the real player stood within 40 map units and 20 degrees of a pose from at least 60 steps earlier (median actual gap: 507 frames, far outside the model's 6-frame context, so nothing but memory can carry the room across). `game` is the same measurement on the real frames: the ceiling, since matching poses are close but never identical.

| config | pairs | model | game (ceiling) | gap | retrieval fired |
|---|---|---|---|---|---|
| sliding context only | 40 | **11.24 +/- 1.57 dB** | 11.25 dB | 0.01 dB | 0/1000 frames |
| memory | 40 | **10.60 +/- 0.63 dB** | 11.25 dB | 0.64 dB | 590/1000 frames |

## What this says

**Retrieval memory does not measurably change return-to-place on this checkpoint.** It moves the score by -0.64 dB against a run-to-run spread of +/-1.69 dB.

*Does the key find the place?* For 36 of the revisits, memory holds at least one frame taken within the revisit tolerance of the current pose. The most similar stored frame is one of them 11% of the time, and one of the top two is 17% of the time. For the other 4 revisits nothing from that place is in memory yet. Keys come from the tokenizer alone, so these rates are the same for every dynamics checkpoint that shares it.

`memory.enabled` is `false` in this config; the `M` key toggles it in `play.py`.

One structural note on the curve: `exclude_recent` blocks retrieval until 64 writes have accumulated (one every 4 frames), so the two configurations are identical by construction for the first few hundred frames.

Regenerate with `python -m ngx.eval.drift --config configs/ladder/26m.yaml --set name=ladder-26m-t4-s0 dynamics_ckpt=runs/ladder-26m-t4-s0/dynamics/final.pt --out docs/DRIFT_26M.md`.
