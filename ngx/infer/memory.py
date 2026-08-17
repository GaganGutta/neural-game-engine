"""Retrieval memory, the drift countermeasure.

The failure this exists to fix: with an 8-frame sliding context, everything the
model knew about a room is gone eight frames after you leave it. Walk out and
walk back and the room is regenerated from scratch -- usually as a different
room. The rollout stays *plausible* while ceasing to be *consistent*, which is
the characteristic way world models fail.

The fix is to give the model back a frame from the last time it stood
somewhere that looked like this, in place of its oldest context slot. Context
length is unchanged, so the model sees exactly the block layout it was trained
on and needs no retraining.

Keys are the bag-of-codes histogram of a frame: which of the tokenizer's
codebook entries appear in it, L2-normalised, compared by cosine similarity.
It costs one 512-wide histogram per frame and works because the tokenizer
already learned to spend different codes on different wall textures -- which is
exactly what distinguishes one room in this maze from another.
"""

from __future__ import annotations

import torch


class RetrievalMemory:
    def __init__(
        self,
        num_codes: int,
        tokens_per_frame: int = 64,
        capacity: int = 4096,
        k: int = 2,
        min_sim: float = 0.9,
        write_every: int = 4,
        exclude_recent: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.num_codes = num_codes
        self.L = tokens_per_frame
        self.capacity = capacity
        self.k = k
        self.min_sim = min_sim
        self.write_every = max(int(write_every), 1)
        # A revisit means somewhere you were a while ago. Without this, the
        # nearest neighbour is always the frame written moments earlier, which
        # tells the model nothing it does not already have in context.
        self.exclude_recent = exclude_recent
        self.reset()

    def reset(self) -> None:
        self.keys = torch.zeros(self.capacity, self.num_codes, device=self.device)
        self.tokens = torch.zeros(self.capacity, self.L, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.stamp = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.size = 0
        self.ptr = 0
        self.clock = 0
        self.steps = 0

    def _key(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(L,)`` token ids -> L2-normalised ``(num_codes,)`` histogram."""
        h = torch.bincount(tokens, minlength=self.num_codes).float()
        return h / h.norm().clamp_min(1e-8)

    def write(self, tokens: torch.Tensor, action: int) -> None:
        self.steps += 1
        if self.steps % self.write_every:
            return
        i = self.ptr
        self.keys[i] = self._key(tokens)
        self.tokens[i] = tokens
        self.actions[i] = int(action)
        self.stamp[i] = self.clock
        self.clock += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def query(self, tokens: torch.Tensor):
        """Top-k stored frames similar to ``tokens``, oldest-first.

        Returns ``(tokens (m, L), actions (m,), sims (m,))`` with ``m <= k``.
        """
        if self.size == 0:
            return None
        key = self._key(tokens)
        sims = self.keys[: self.size] @ key
        fresh = self.stamp[: self.size] > (self.clock - self.exclude_recent)
        sims = sims.masked_fill(fresh, -1.0)

        m = min(self.k, int((sims >= self.min_sim).sum()))
        if m == 0:
            return None
        idx = sims.topk(m).indices
        # Oldest first, so the retrieved frames read as history rather than as
        # something that just happened.
        idx = idx[self.stamp[idx].argsort()]
        return self.tokens[idx], self.actions[idx], sims[idx]

    def augment(self, tokens: torch.Tensor, actions: torch.Tensor):
        """Swap retrieved frames into the oldest context slots.

        ``tokens`` is ``(C, L)`` and ``actions`` is ``(C,)``. Returns the same
        shapes plus how many slots were replaced, so callers can show it.
        """
        hit = self.query(tokens[-1])
        if hit is None:
            return tokens, actions, 0
        r_tok, r_act, _ = hit
        m = min(len(r_tok), tokens.shape[0] - 1)  # never displace the current frame
        if m == 0:
            return tokens, actions, 0
        tokens = tokens.clone()
        actions = actions.clone()
        tokens[:m] = r_tok[:m]
        actions[:m] = r_act[:m]
        return tokens, actions, m
