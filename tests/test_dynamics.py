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


def _model(seed: int = 0) -> DynamicsTransformer:
    torch.manual_seed(seed)
    m = DynamicsTransformer(
        num_codes=NUM_CODES, num_actions=NUM_ACTIONS, tokens_per_frame=L,
        context=CONTEXT, d_model=64, n_layers=3, n_heads=4,
    )
    return m.eval()


def test_cached_inference_matches_training_forward():
    m = _model()
    B, T = 2, CONTEXT + 1
    torch.manual_seed(1)
    tokens = torch.randint(0, NUM_CODES, (B, T, L))
    actions = torch.randint(0, NUM_ACTIONS, (B, T))

    # Fully mask the final target frame; leave earlier targets fully masked too
    # so the comparison isolates the last block.
    mask = torch.ones(B, T - 1, L, dtype=torch.bool)
    with torch.no_grad():
        train_logits = m(tokens, actions, mask)[:, -1]  # (B, L, V)

    # Inference path: prefix is frames 0..T-2 with their actions.
    with torch.no_grad():
        cache = m.encode_prefix(tokens[:, : T - 1], actions[:, : T - 1])
        target = torch.full((B, L), m.mask_token, dtype=torch.long)
        infer_logits = m.decode_logits(cache, target, frame_index=T - 1)

    err = (train_logits - infer_logits).abs().max().item()
    assert err < 1e-4, f"train/infer logits diverge by {err}"


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
