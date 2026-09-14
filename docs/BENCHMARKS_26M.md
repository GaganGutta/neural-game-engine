# Benchmarks

- device: `cpu` (AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD)
- torch: `2.10.0+cpu`, platform: `Windows-11-10.0.26200-SP0`
- model: 25.7M params, context 6 frames, 64 tokens/frame
- greedy decoding, 3 warmup frames discarded, up to 30 timed frames per row (cap 90s)

Each row applies one change on top of the fastest configuration so far. A change that measures slower is reverted, and says so.

`output delta` is PSNR between this row's rollout and the rollout of the configuration the change was applied to. It answers one question: *did this change alter what comes out?* `identical` means the transformation is exact. A finite number means the output moved -- which is expected when the decoder itself changes, and a warning sign when only the numerics did. It is not a quality score; for decode quality against ground truth see [DECODE.md](DECODE.md).

| step | fps | ms/frame | passes/frame | vs. row 1 | weights | peak mem | output delta | |
|---|---|---|---|---|---|---|---|---|
| raster AR, no KV cache | **0.09** | 10595.9 | 64 | 1.0x | 102.9 MB | 496 MB | identical | kept |
| + KV cache (within frame) | **0.60** | 1668.4 | 64 | 6.4x | 102.9 MB | 510 MB | identical | kept |
| + MaskGIT parallel decode | **5.55** | 180.0 | 4 | 58.8x | 102.9 MB | 504 MB | 13.7 dB | kept |
| + carry KV cache across frames | **7.87** | 127.0 | 4 | 83.4x | 102.9 MB | 519 MB | 16.1 dB | kept |
| + bf16 autocast | **9.29** | 107.7 | 4 | 98.4x | 102.9 MB | 576 MB | 14.2 dB | kept |
| + torch.compile | unavailable | | | | | | | _RuntimeError: Compiler: cl is not found._ |
| + int8 dynamic quant | **10.56** | 94.7 | 4 | 111.9x | 26.7 MB | 848 MB | 12.1 dB | kept |

`peak mem` is process RSS on CPU and peak allocated VRAM on CUDA; on CPU it includes the interpreter and both models, so treat it as an envelope rather than a model footprint. `weights` is the dynamics model's parameter bytes.

`identical` on the KV-cache row is the point of that row. Caching the prefix is an exact transformation, and a bit-for-bit identical rollout under greedy decoding is the proof rather than the claim.

Regenerate with `python -m ngx.eval.bench --config configs/ladder/26m.yaml --set name=ladder-26m-t4-s0 dynamics_ckpt=runs/ladder-26m-t4-s0/dynamics/final.pt --out docs/BENCHMARKS_26M.md`.
