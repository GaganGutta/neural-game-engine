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

Keys are the mean of the *codebook embedding vectors* of a frame's tokens,
L2-normalised and compared by cosine similarity. The obvious alternative -- a
bag-of-codes histogram over the 512 codebook entries -- was tried first and
measured worse in the way that mattered: it ranks about as well, but two frames
of the same wall from slightly different angles land on different-but-adjacent
codes, which a histogram scores as *no* overlap at all. Measured on the
reference trajectory, matched revisits averaged 0.34 similarity against 0.09
for random pairs, so any threshold high enough to sound like "similar" fired on
nothing. Averaging the embeddings instead uses the metric structure the
codebook already learned: revisits average 0.96 against 0.47 for random pairs,
and a 0.9 threshold means what it looks like it means.

Mean-pooling over the whole frame, rather than pooling spatially, is also
measured: a 2x2 spatial key dropped top-1 retrieval accuracy from 0.55 to 0.14,
because turning your head moves content across the grid and a spatially-aware
key reads that as a different place.
"""

from __future__ import annotations

import torch


class RetrievalMemory:
    def __init__(
        self,
        code_embed: torch.Tensor,
        tokens_per_frame: int = 64,
        capacity: int = 4096,
        k: int = 2,
        min_sim: float = 0.9,
        write_every: int = 4,
        exclude_recent: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        # The tokenizer's codebook, (num_codes, D). Frozen; used only to turn
        # token ids back into vectors that can be averaged.
        self.E = code_embed.detach().to(self.device).float()
        self.dim = self.E.shape[1]
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
        self.keys = torch.zeros(self.capacity, self.dim, device=self.device)
        self.tokens = torch.zeros(self.capacity, self.L, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.stamp = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.size = 0
        self.ptr = 0
        self.clock = 0
        self.steps = 0

    def _key(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(L,)`` token ids -> L2-normalised mean codebook embedding."""
        v = self.E[tokens].mean(0)
        return v / v.norm().clamp_min(1e-8)

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
