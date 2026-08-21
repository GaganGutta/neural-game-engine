# Cache carrying across frame boundaries: exactness chosen as the default

Under rope, extending the KV cache by one block is exactly equal to recomputing
the whole prefix (asserted to 1e-7 in `tests/test_dynamics.py`). What is not
exact is eviction: when the window slides, every retained block still carries
history it computed while the now-dropped block was visible, which a fresh
recompute would not give it. Measured on rollouts from matched starts with
greedy decoding (6 starts x 60 frames, untrained rope model of the
`configs/small.yaml` shape), never-refreshed carrying first *diverged* from
full recompute at frame 5 and ended with 40% of frames byte-identical, for a
1.25x speedup (32.7 vs 40.7 ms/frame). To be precise about what that shows:
divergence, not degradation. Two autoregressive rollouts separate after any
perturbation, and nothing here established that the carried rollout is worse --
it is a different plausible rollout from the same model. What carrying costs is
*reproducibility*: the rollout you get is no longer the one a from-scratch
computation defines, and every evaluation in this repo compares rollouts frame
by frame against references. **We chose exactness as the default**
(`carry_cache: false`, full rebuild at every frame boundary); carrying remains
a single benchmark row in [BENCHMARKS.md](BENCHMARKS.md) for rope checkpoints,
and whether its 1.25x is worth buying is a quality judgement that would need a
side-by-side comparison nobody has run.
