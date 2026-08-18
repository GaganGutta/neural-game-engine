"""Engine and memory behaviour, on an untrained model.

None of these need a trained network -- they check that the machinery around
it is sound, which is where the bugs that survive a training run actually live.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngx.infer.engine import EngineConfig, NeuralGameEngine  # noqa: E402
from ngx.infer.memory import RetrievalMemory  # noqa: E402
from ngx.models.dynamics import DynamicsTransformer  # noqa: E402
from ngx.models.vqvae import VQVAE  # noqa: E402

NUM_CODES = 64
NUM_ACTIONS = 6
CONTEXT = 4


def _engine(**cfg_kw) -> NeuralGameEngine:
    torch.manual_seed(0)
    vq = VQVAE(ch=16, embed_dim=8, num_codes=NUM_CODES, n_res=1)
    dyn = DynamicsTransformer(
        num_codes=NUM_CODES, num_actions=NUM_ACTIONS, tokens_per_frame=vq.tokens_per_frame,
        context=CONTEXT, d_model=32, n_layers=2, n_heads=4,
    )
    cfg = EngineConfig(temperature=0.0, top_k=0, **cfg_kw)
    return NeuralGameEngine(vq, dyn, cfg, device="cpu")


def _seed_frames(n=CONTEXT):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8)


def test_step_returns_a_frame_and_slides_the_window():
    e = _engine()
    e.reset(_seed_frames())
    before = e.tokens.clone()
    frame = e.step(3)
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    # Context slid by one: old rows shift down, newest row is the prediction.
    assert torch.equal(e.tokens[:-1], before[1:])


def test_kv_cache_is_numerically_exact():
    """Within-frame caching is an optimisation, not an approximation.

    Carrying across frames is disabled here on purpose: that path evicts the
    oldest block and is a measured approximation, covered separately in
    tests/test_dynamics.py. This test is about the claim the benchmark table
    makes, which is that caching the prefix inside one frame changes nothing.
    """
    seeds = _seed_frames()
    outs = []
    for use_cache in (False, True):
        torch.manual_seed(7)
        e = _engine(use_cache=use_cache, carry_cache=False, decode="maskgit", maskgit_steps=4)
        e.reset(seeds)
        outs.append(np.stack([e.step(a) for a in (3, 1, 3, 2)]))
    assert np.array_equal(outs[0], outs[1]), "KV cache changed the output"


def test_carrying_the_cache_across_frames_does_not_change_the_rollout():
    """End-to-end check that the eviction approximation is invisible in play.

    Greedy decoding, so any divergence would show up as a different frame
    rather than as sampling noise.
    """
    seeds = _seed_frames()
    outs = []
    for carry in (False, True):
        torch.manual_seed(11)
        e = _engine(carry_cache=carry, decode="maskgit", maskgit_steps=4)
        assert e._carry == carry, "carry flag did not take effect"
        e.reset(seeds)
        outs.append(np.stack([e.step(a) for a in (3, 3, 1, 3, 2, 3, 3, 1)]))
    same = float(np.mean(outs[0] == outs[1]))
    assert same > 0.999, f"carrying changed {100 * (1 - same):.2f}% of pixels"


def test_decoders_emit_only_real_codebook_entries():
    """A leftover MASK token would index past the codebook and crash decoding."""
    for decode in ("maskgit", "raster"):
        e = _engine(decode=decode, maskgit_steps=4)
        e.reset(_seed_frames())
        e.step(3)
        assert int(e.tokens[-1].max()) < NUM_CODES, f"{decode} left a mask token behind"
        assert int(e.tokens[-1].min()) >= 0


def test_maskgit_uses_far_fewer_passes_than_raster():
    calls = {"maskgit": 0, "raster": 0}
    for decode in calls:
        e = _engine(decode=decode, maskgit_steps=8)
        e.reset(_seed_frames())
        real = e._decode_logits

        def counted(*args, _d=decode, _f=real, **kw):
            calls[_d] += 1
            return _f(*args, **kw)

        e._decode_logits = counted
        e.step(3)
    assert calls["raster"] == e.L
    assert calls["maskgit"] == 8
    assert calls["maskgit"] < calls["raster"]


# -- retrieval memory ------------------------------------------------------
def _mem(**kw):
    kw.setdefault("write_every", 1)
    kw.setdefault("exclude_recent", 3)
    kw.setdefault("min_sim", 0.9)
    torch.manual_seed(0)
    # Stand-in codebook: near-orthogonal rows, so unrelated frames score low.
    embed = torch.randn(NUM_CODES, 32)
    return RetrievalMemory(code_embed=embed, tokens_per_frame=8, capacity=32, k=2, **kw)


def test_memory_recalls_the_matching_frame_not_the_recent_one():
    m = _mem()
    room_a = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    room_b = torch.tensor([9, 9, 10, 10, 11, 11, 12, 12])
    m.write(room_a, 3)
    for _ in range(8):  # wander through a different room
        m.write(room_b, 1)
    hit = m.query(room_a)
    assert hit is not None, "did not recall a room it had seen"
    toks, acts, sims = hit
    assert torch.equal(toks[0], room_a) and int(acts[0]) == 3
    assert float(sims[0]) > 0.99


def test_memory_ignores_the_immediate_past():
    """Retrieving the frame written moments ago adds nothing to the context."""
    m = _mem(exclude_recent=10)
    f = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    for _ in range(5):
        m.write(f, 0)
    assert m.query(f) is None


def test_memory_declines_when_nothing_is_similar_enough():
    m = _mem()
    m.write(torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]), 0)
    for _ in range(5):
        m.write(torch.tensor([30, 31, 32, 33, 34, 35, 36, 37]), 0)
    assert m.query(torch.tensor([50, 51, 52, 53, 54, 55, 56, 57])) is None


def test_augment_keeps_context_length_and_spares_the_current_frame():
    m = _mem()
    old = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    m.write(old, 5)
    for _ in range(8):
        m.write(torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]), 1)

    tokens = torch.stack([torch.full((8,), 7) for _ in range(CONTEXT)])
    tokens[-1] = old  # the player is back in the old room
    actions = torch.zeros(CONTEXT, dtype=torch.long)

    new_tok, new_act, n = m.augment(tokens, actions)
    assert n >= 1, "memory did not fire on a revisit"
    assert new_tok.shape == tokens.shape, "context length changed"
    assert torch.equal(new_tok[-1], old), "current frame was overwritten"
    assert torch.equal(new_tok[0], old), "retrieved frame not placed in the oldest slot"
    assert int(new_act[0]) == 5, "retrieved frame lost its action"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
