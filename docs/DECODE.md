# Decoding passes vs quality

One-step prediction on 40 held-out windows: seed the model with real frames, apply the real action, compare the predicted frame to the frame the game actually drew. No rollout, so this isolates decode quality from drift. Greedy sampling throughout.

| decoder | passes | PSNR vs real next frame | sharpness vs real | ms/frame |
|---|---|---|---|---|
| MaskGIT, 1 passes | 1 | 24.76 dB | **0.51x** | 13.7 |
| MaskGIT, 2 passes | 2 | 24.34 dB | **0.51x** | 17.3 |
| MaskGIT, 4 passes | 4 | 24.35 dB | **0.53x** | 23.2 |
| MaskGIT, 8 passes | 8 | 24.50 dB | **0.53x** | 36.9 |
| MaskGIT, 16 passes | 16 | 24.08 dB | **0.52x** | 63.7 |
| MaskGIT, 32 passes | 32 | 23.90 dB | **0.52x** | 112.7 |
| raster AR, 64 passes | 64 | 23.35 dB | **0.52x** | 206.0 |

## What this says

**Extra decoding passes buy nothing here.** PSNR drifts slightly *down* as passes increase and sharpness is flat to two decimal places. Raster spends 64 passes and 14x the latency of a single pass to end up marginally worse on both.

The mild PSNR decline is expected and is not evidence that fewer passes produce better pictures. One pass takes an independent argmax at every position -- the per-token conditional mode -- which is smooth, and smooth wins at PSNR. More passes commit tokens and condition on them, producing something closer to a sample than to an average: further from the mean, so lower PSNR.

The sharpness column is what rules out the flattering reading. A ratio of 1.0 would mean the prediction carries as much fine detail as the frame the game drew. Every row sits near 0.5, and every row sits there *equally*. The blur is not something the decoder can fix, because it is not coming from the decoder -- at 2.0M parameters and under one pass over the data, the model is the bottleneck and the decoding schedule is not.

So the honest conclusion is narrow: **at this scale**, MaskGIT at a low pass count dominates raster on latency at no cost in quality, and spending more passes is latency bought for nothing. That is why `infer.maskgit_steps` is 4 rather than a rounder-sounding 8. It is not a claim about MaskGIT in general -- with a model big enough for the decoder to be the limiting factor, this table would look different, and it should be re-run before assuming otherwise.

Timings use greedy decoding so the rows are comparable. `play.py` samples (`temperature`, `top_k`), which trades a little of this PSNR for variety.

Regenerate with `python -m ngx.eval.decode_quality --config configs/small.yaml`.
