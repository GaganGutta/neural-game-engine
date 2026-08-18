"""The invariant everything else rests on.

``play.py`` never runs the training forward pass. It runs ``encode_prefix``
once and then ``decode_logits`` per iteration against the cached keys and
values. If those two paths ever disagree, the model you play is not the model
you trained -- and the failure is silent, because a slightly-wrong world model
still produces plausible-looking Doom.

So: assert they produce identical logits, and assert the attention mask has the
exact shape the docstring in dynamics.py claims.

    python -m pytest tests/ -q          (or just: python tests/test_dynamics.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngx.models.dynamics import DynamicsTransformer  # noqa: E402

L = 16  # smaller than the real 64 so the test is quick
CONTEXT = 4
NUM_CODES = 32
NUM_ACTIONS = 6


def _model(seed: int = 0, pos_encoding: str = "rope") -> DynamicsTransformer:
    torch.manual_seed(seed)
    m = DynamicsTransformer(
        num_codes=NUM_CODES, num_actions=NUM_ACTIONS, tokens_per_frame=L,
        context=CONTEXT, d_model=64, n_layers=3, n_heads=4, pos_encoding=pos_encoding,
    )
    return m.eval()


def test_cached_inference_matches_training_forward():
    for pos in ("rope", "absolute"):
        m = _model(pos_encoding=pos)
        B, T = 2, CONTEXT + 1
        torch.manual_seed(1)
        tokens = torch.randint(0, NUM_CODES, (B, T, L))
        actions = torch.randint(0, NUM_ACTIONS, (B, T))

        # Fully mask the final target frame; leave earlier targets fully masked
        # too so the comparison isolates the last block.
        mask = torch.ones(B, T - 1, L, dtype=torch.bool)
        with torch.no_grad():
            train_logits = m(tokens, actions, mask)[:, -1]  # (B, L, V)
            cache = m.encode_prefix(tokens[:, : T - 1], actions[:, : T - 1])
            target = torch.full((B, L), m.mask_token, dtype=torch.long)
            infer_logits = m.decode_logits(cache, target, frame_index=T - 1)

        err = (train_logits - infer_logits).abs().max().item()
        assert err < 1e-4, f"[{pos}] train/infer logits diverge by {err}"


def test_carried_cache_is_exact_when_nothing_is_evicted():
    """Extending a rope cache by one block equals recomputing the whole prefix.

    This is the property rope buys. Under absolute positions every block's
    embedding is indexed by its slot in the window, so a slide invalidates the
    entire cache; under rope a cached key keeps its own absolute rotation and a
    later query only ever sees the relative offset.
    """
    m = _model()
    torch.manual_seed(4)
    tokens = torch.randint(0, NUM_CODES, (1, CONTEXT + 1, L))
    actions = torch.randint(0, NUM_ACTIONS, (1, CONTEXT + 1))
    with torch.no_grad():
        part = m.encode_prefix(tokens[:, :CONTEXT], actions[:, :CONTEXT], base=0)
        carried = m.extend_prefix(part, tokens[:, CONTEXT:], actions[:, CONTEXT:], base=CONTEXT)
        fresh = m.encode_prefix(tokens, actions, base=0)

        kerr = max((a[0] - b[0]).abs().max().item() for a, b in zip(carried, fresh))
        verr = max((a[1] - b[1]).abs().max().item() for a, b in zip(carried, fresh))
        target = torch.full((1, L), m.mask_token, dtype=torch.long)
        lerr = (m.decode_logits(carried, target, CONTEXT + 1)
                - m.decode_logits(fresh, target, CONTEXT + 1)).abs().max().item()
    assert kerr < 1e-4 and verr < 1e-4, f"cached K/V drift: {kerr}, {verr}"
    assert lerr < 1e-4, f"carried-cache logits diverge by {lerr}"


def test_eviction_is_an_approximation_and_a_small_one():
    """Dropping the oldest block is *not* exact, and this pins how inexact.

    A retained block's cached representation was computed while the evicted
    block was still visible; a fresh window would give it no such history. No
    position encoding fixes that -- it is a property of block-causal attention,
    not of rope. So the test asserts what actually matters at play time: the
    approximation must not change which token is chosen.
    """
    m = _model()
    torch.manual_seed(5)
    tokens = torch.randint(0, NUM_CODES, (1, CONTEXT + 1, L))
    actions = torch.randint(0, NUM_ACTIONS, (1, CONTEXT + 1))
    step = L + 1
    with torch.no_grad():
        part = m.encode_prefix(tokens[:, :CONTEXT], actions[:, :CONTEXT], base=0)
        carried = m.extend_prefix(part, tokens[:, CONTEXT:], actions[:, CONTEXT:], base=CONTEXT)
        evicted = [(k[:, :, step:], v[:, :, step:]) for k, v in carried]
        fresh = m.encode_prefix(tokens[:, 1:], actions[:, 1:], base=1)

        target = torch.full((1, L), m.mask_token, dtype=torch.long)
        le = m.decode_logits(evicted, target, CONTEXT + 1)
        lr = m.decode_logits(fresh, target, CONTEXT + 1)

    rel = (le - lr).abs().max().item() / lr.std().item()
    agree = (le.argmax(-1) == lr.argmax(-1)).float().mean().item()
    assert agree == 1.0, f"eviction changed {100 * (1 - agree):.1f}% of argmax tokens"
    assert rel < 0.1, f"eviction perturbs logits by {rel:.3f} of a standard deviation"


def test_future_frames_cannot_leak_into_a_prediction():
    """Changing frame s must not change the prediction for frame s."""
    m = _model()
    B, T = 1, CONTEXT + 1
    torch.manual_seed(2)
    tokens = torch.randint(0, NUM_CODES, (B, T, L))
    actions = torch.randint(0, NUM_ACTIONS, (B, T))
    mask = torch.ones(B, T - 1, L, dtype=torch.bool)

    with torch.no_grad():
        base = m(tokens, actions, mask)
        # Scramble the last frame's ground truth. Its own prediction, and every
        # earlier one, must be untouched.
        altered = tokens.clone()
        altered[:, -1] = (altered[:, -1] + 7) % NUM_CODES
        after = m(altered, actions, mask)
    assert torch.allclose(base, after, atol=1e-6), "target frame leaked into its own prediction"

    with torch.no_grad():
        # The action at the final frame is applied *after* the last prediction,
        # so it must not matter either.
        alt_a = actions.clone()
        alt_a[:, -1] = (alt_a[:, -1] + 1) % NUM_ACTIONS
        after_a = m(tokens, alt_a, mask)
    assert torch.allclose(base, after_a, atol=1e-6), "future action leaked"


def test_action_actually_conditions_the_prediction():
    """A world model that ignores the controller is not a game engine."""
    m = _model()
    B, T = 1, CONTEXT + 1
    torch.manual_seed(3)
    tokens = torch.randint(0, NUM_CODES, (B, T, L))
    actions = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T - 1, L, dtype=torch.bool)

    with torch.no_grad():
        base = m(tokens, actions, mask)[:, -1]
        alt = actions.clone()
        alt[:, -2] = 3  # the action that produces the final frame
        after = m(tokens, alt, mask)[:, -1]
    delta = (base - after).abs().max().item()
    assert delta > 1e-5, "prediction is invariant to the action that caused it"


def test_mask_shape_and_visibility_counts():
    m = _model()
    T = CONTEXT + 1
    S = T - 1
    n_a, n_b = S * (L + 1), S * L
    assert m.train_mask.shape == (n_a + n_b, n_a + n_b)

    # First target block sees exactly one A block plus its own L tokens.
    first_b = n_a
    assert int(m.train_mask[first_b].sum()) == (L + 1) + L
    # Last target block sees every A block plus its own.
    last_b = n_a + (S - 1) * L
    assert int(m.train_mask[last_b].sum()) == S * (L + 1) + L
    # Stream A never attends to stream B.
    assert not m.train_mask[:n_a, n_a:].any()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
