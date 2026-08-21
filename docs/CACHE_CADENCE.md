# Cache carrying across frame boundaries: measured, and off

Under rope, extending the KV cache by one block is exactly equal to recomputing
the whole prefix (asserted to 1e-7 in `tests/test_dynamics.py`). What is not
exact is eviction: when the window slides, every retained block still carries
history it computed while the now-dropped block was visible, which a fresh
recompute would not give it. Measured on rollouts from matched starts with
greedy decoding (6 starts x 60 frames, untrained rope model of the
`configs/small.yaml` shape), never-refreshed carrying first diverged from full
recompute at frame 5 and ended with only 40% of frames byte-identical, in
exchange for a 1.25x speedup (32.7 vs 40.7 ms/frame); intermediate refresh
cadences bought back little, because once one token differs the rollouts have
different context and nothing recovers them. That divergence is material, so
**the default is a full cache rebuild at every frame boundary, which is
exact** (`carry_cache: false`). Carrying remains a single benchmark row in
[BENCHMARKS.md](BENCHMARKS.md) for rope checkpoints. One caveat carried
forward: these divergence numbers come from an untrained model, and a second
random init of the same shape did not diverge at all in twelve frames, so the
row is worth re-reading on the first trained rope checkpoint before treating
1.25x as forever lost.
