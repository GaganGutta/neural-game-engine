# Benchmarks

- device: `cpu` (AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD)
- torch: `2.10.0+cpu`, platform: `Windows-11-10.0.26200-SP0`
- model: 2.0M params, context 6 frames, 64 tokens/frame
- greedy decoding, 3 warmup frames discarded, up to 30 timed frames per row (cap 90s)

Each row applies one change on top of the fastest configuration so far. A change that measures slower is reverted, and says so.

`output delta` is PSNR between this row's rollout and the rollout of the configuration the change was applied to. It answers one question: *did this change alter what comes out?* `identical` means the transformation is exact. A finite number means the output moved -- which is expected when the decoder itself changes, and a warning sign when only the numerics did. It is not a quality score; for decode quality against ground truth see [DECODE.md](DECODE.md).

| step | fps | ms/frame | passes/frame | vs. row 1 | weights | peak mem | output delta | |
|---|---|---|---|---|---|---|---|---|
| raster AR, no KV cache | **0.65** | 1540.6 | 64 | 1.0x | 8.0 MB | 286 MB | identical | kept |
| + KV cache (within frame) | **4.11** | 243.3 | 64 | 6.3x | 8.0 MB | 282 MB | identical | kept |
| + MaskGIT parallel decode | **35.47** | 28.2 | 4 | 54.7x | 8.0 MB | 292 MB | 14.3 dB | kept |
| + carry KV cache across frames | unavailable | | | | | | | _checkpoint uses absolute positions_ |
| + bf16 autocast | **27.18** | 36.8 | 4 | 41.9x | 8.0 MB | 295 MB | 16.2 dB | reverted |
| + torch.compile | unavailable | | | | | | | _RuntimeError: Compiler: cl is not found._ |
| + int8 dynamic quant | **28.72** | 34.8 | 4 | 44.2x | 2.4 MB | 441 MB | 14.6 dB | reverted |

`peak mem` is process RSS on CPU and peak allocated VRAM on CUDA; on CPU it includes the interpreter and both models, so treat it as an envelope rather than a model footprint. `weights` is the dynamics model's parameter bytes.

`identical` on the KV-cache row is the point of that row. Caching the prefix is an exact transformation, and a bit-for-bit identical rollout under greedy decoding is the proof rather than the claim.

Regenerate with `python -m ngx.eval.bench --config configs/small.yaml`.
