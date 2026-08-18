"""Action-conditioned dynamics model.

Sequence layout
---------------
A window of ``T = context + 1`` frames becomes two streams.

*Stream A* (clean context), one block per frame::

    [ f_t[0] .. f_t[L-1] ][ a_t ]        block length L + 1

where ``a_t`` is the action applied at frame ``t``. Putting the action last in
the block means "everything before block ``t+1``" is exactly the information a
player has when frame ``t+1`` is about to be drawn.

*Stream B* (masked prediction targets), one block per predicted frame::

    [ ~f_s[0] .. ~f_s[L-1] ]             block length L

Attention
---------
* A block ``t`` sees A blocks ``0..t``, bidirectionally inside its own block.
* B block ``s`` sees A blocks ``0..s-1`` and itself. Nothing else.

The two streams exist so that context is always *clean*. If we instead masked
tokens in place, later frames would attend to a corrupted history that never
occurs at inference. Duplicating the targets costs ~2x sequence length and buys
``T-1`` supervised frames per forward pass instead of one.

At inference stream B holds exactly one block, so its attention is "everything
in the prefix, plus itself" -- no mask at all, and the prefix's keys and values
are computed once and reused across every decoding iteration.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope(pos: torch.Tensor, d_head: int, base: float = 10000.0, device=None):
    """Rotary tables for the given absolute positions.

    Returns ``(cos, sin)`` of shape ``(N, d_head)``, ready to broadcast over
    ``(B, H, N, d_head)``.
    """
    half = d_head // 2
    freq = base ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    ang = pos.to(device=device, dtype=torch.float32)[:, None] * freq[None, :]
    return torch.cat([ang.cos()] * 2, dim=-1), torch.cat([ang.sin()] * 2, dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``(B, H, N, D)`` by the per-position angles in ``cos``/``sin``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if rope is not None:
            # Rotate before caching, so cached keys carry their own absolute
            # position and a later query only ever sees the relative offset.
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        kv = (k, v) if return_kv else None
        if prefix_kv is not None:
            # Queries are the target block; keys/values are the cached prefix
            # followed by the target block itself.
            k = torch.cat([prefix_kv[0], k], dim=2)
            v = torch.cat([prefix_kv[1], v], dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)
        return (out, kv) if return_kv else out


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x, attn_mask=None, prefix_kv=None, return_kv=False, rope=None):
        h = self.ln1(x)
        if return_kv:
            a, kv = self.attn(h, attn_mask, prefix_kv, return_kv=True, rope=rope)
        else:
            a, kv = self.attn(h, attn_mask, prefix_kv, rope=rope), None
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return (x, kv) if return_kv else x


class DynamicsTransformer(nn.Module):
    def __init__(
        self,
        num_codes: int,
        num_actions: int,
        tokens_per_frame: int = 64,
        context: int = 8,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
        pos_encoding: str = "rope",
    ) -> None:
        super().__init__()
        if pos_encoding not in ("rope", "absolute"):
            raise ValueError(f"pos_encoding must be 'rope' or 'absolute', got {pos_encoding!r}")
        self.pos_encoding = pos_encoding
        self.num_codes = num_codes
        self.mask_token = num_codes  # one id past the codebook
        self.num_actions = num_actions
        self.L = tokens_per_frame
        self.context = context
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.tok_emb = nn.Embedding(num_codes + 1, d_model)
        self.act_emb = nn.Embedding(num_actions, d_model)
        # The two schemes own disjoint parameters, so a checkpoint from one
        # cannot be loaded into the other by accident.
        if pos_encoding == "absolute":
            # Position inside a block (L frame slots plus one action slot), and
            # which frame of the window this block is. Both are indexed by slot,
            # which is exactly why a cache cannot survive the window sliding.
            self.pos_emb = nn.Embedding(self.L + 1, d_model)
            self.frame_emb = nn.Embedding(context + 2, d_model)
        else:
            # Rope carries position in the rotation. What it cannot carry is
            # "this slot holds an action rather than a frame token" -- under the
            # absolute scheme pos_emb[L] said that implicitly, and rope encodes
            # only distance, so it has to be said out loud.
            self.type_emb = nn.Embedding(2, d_model)
        # Clean-context vs prediction-target stream.
        self.seg_emb = nn.Embedding(2, d_model)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_codes, bias=False)

        self.apply(self._init)
        # Shrink residual-path outputs so activations do not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

        self.register_buffer(
            "train_mask", self._build_train_mask(context + 1), persistent=False
        )

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # -- masks --------------------------------------------------------------
    def _build_train_mask(self, T: int) -> torch.Tensor:
        """``(N, N)`` bool, True where a query may attend to a key."""
        L = self.L
        S = T - 1  # A blocks 0..T-2 and B blocks 1..T-1
        # A block T-1 is deliberately absent: no target ever attends to it.
        a_block = torch.arange(S).repeat_interleave(L + 1)
        b_block = torch.arange(1, T).repeat_interleave(L)
        blk = torch.cat([a_block, b_block])
        is_a = torch.cat([torch.ones_like(a_block), torch.zeros_like(b_block)]).bool()

        qb, kb = blk[:, None], blk[None, :]
        qa, ka = is_a[:, None], is_a[None, :]
        return torch.where(
            qa,
            ka & (kb <= qb),                 # A sees A up to and including itself
            (ka & (kb <= qb - 1)) | (~ka & (kb == qb)),  # B sees earlier A, and itself
        )

    # -- training -----------------------------------------------------------
    def forward(self, tokens: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        """``tokens`` ``(B, T, L)``, ``actions`` ``(B, T)``, ``mask`` ``(B, T-1, L)``.

        ``mask[:, s-1]`` marks the positions of frame ``s`` to hide and predict.
        Returns logits ``(B, T-1, L, num_codes)``.
        """
        B, T, L = tokens.shape
        S = T - 1
        dev = tokens.device

        # Stream A: frames 0..T-2 with their actions, always clean.
        a_tok = self.tok_emb(tokens[:, :S])                       # (B,S,L,D)
        a_act = self.act_emb(actions[:, :S]).unsqueeze(2)         # (B,S,1,D)
        a = torch.cat([a_tok, a_act], dim=2)                      # (B,S,L+1,D)
        a = a + self._type_bias(L + 1, dev)
        a = a + self.seg_emb.weight[0]

        # Stream B: masked copies of frames 1..T-1.
        tgt = torch.where(mask, self.mask_token, tokens[:, 1:])    # (B,S,L)
        b = self.tok_emb(tgt)
        b = b + self._type_bias(L, dev)
        b = b + self.seg_emb.weight[1]

        if self.pos_encoding == "absolute":
            a = a + self.pos_emb.weight[: L + 1]
            a = a + self.frame_emb(torch.arange(S, device=dev))[None, :, None]
            b = b + self.pos_emb.weight[:L]
            b = b + self.frame_emb(torch.arange(1, T, device=dev))[None, :, None]

        x = torch.cat([a.reshape(B, S * (L + 1), -1), b.reshape(B, S * L, -1)], dim=1)
        rope = None
        if self.pos_encoding == "rope":
            pos = torch.cat([self.a_positions(S, dev), self.b_positions(S, dev)])
            rope = build_rope(pos, self.d_head, device=dev)

        attn = self.train_mask[None, None]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn, rope=rope)
        x = self.ln_f(x[:, S * (L + 1) :])
        return self.head(x).view(B, S, L, self.num_codes)

    # -- positions ----------------------------------------------------------
    def _type_bias(self, width: int, device):
        """``(1, 1, width, D)``: flag the action slot when the block has one.

        A no-op under the absolute scheme, where ``pos_emb`` already gives the
        action slot its own vector.
        """
        if self.pos_encoding == "absolute":
            return 0.0
        ids = torch.zeros(width, dtype=torch.long, device=device)
        if width == self.L + 1:
            ids[-1] = 1
        return self.type_emb(ids)[None, None]

    def a_positions(self, n_blocks: int, device, base: int = 0) -> torch.Tensor:
        """Rope positions for ``n_blocks`` stream-A blocks starting at frame ``base``."""
        t = torch.arange(n_blocks, device=device).repeat_interleave(self.L + 1)
        j = torch.arange(self.L + 1, device=device).repeat(n_blocks)
        return (base + t) * (self.L + 1) + j

    def b_positions(self, n_blocks: int, device, base: int = 0) -> torch.Tensor:
        """A target block sits at the positions its frame would occupy in stream A."""
        s = torch.arange(1, n_blocks + 1, device=device).repeat_interleave(self.L)
        j = torch.arange(self.L, device=device).repeat(n_blocks)
        return (base + s) * (self.L + 1) + j

    def sample_mask(self, B: int, S: int, L: int, device) -> torch.Tensor:
        """Cosine masking schedule: hide ``cos(u*pi/2)`` of each target frame."""
        u = torch.rand(B, S, device=device)
        n_mask = (torch.cos(u * math.pi / 2) * L).ceil().clamp(1, L).long()
        rank = torch.rand(B, S, L, device=device).argsort(-1).argsort(-1)
        return rank < n_mask.unsqueeze(-1)

    def loss(self, tokens: torch.Tensor, actions: torch.Tensor):
        mask = self.sample_mask(tokens.shape[0], tokens.shape[1] - 1, self.L, tokens.device)
        logits = self(tokens, actions, mask)
        target = tokens[:, 1:]
        sel = mask
        loss = F.cross_entropy(logits[sel], target[sel])
        with torch.no_grad():
            acc = (logits[sel].argmax(-1) == target[sel]).float().mean().item()
        return loss, {"loss": loss.item(), "token_acc": acc}

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def encode_prefix(self, tokens: torch.Tensor, actions: torch.Tensor, base: int = 0):
        """Run stream A once and keep its keys and values.

        ``tokens`` ``(B, C, L)`` and ``actions`` ``(B, C)`` for the C context
        frames. The returned cache stays valid for every decoding iteration of
        the next frame, which is where the speedup in the benchmark table comes
        from.
        """
        B, C, L = tokens.shape
        device = tokens.device
        a = torch.cat(
            [self.tok_emb(tokens), self.act_emb(actions).unsqueeze(2)], dim=2
        )
        a = a + self._type_bias(L + 1, device)
        a = a + self.seg_emb.weight[0]
        if self.pos_encoding == "absolute":
            a = a + self.pos_emb.weight[: L + 1]
            a = a + self.frame_emb(torch.arange(C, device=device))[None, :, None]
        x = a.reshape(B, C * (L + 1), -1)

        rope = None
        if self.pos_encoding == "rope":
            rope = build_rope(self.a_positions(C, device, base), self.d_head, device=device)

        blk_id = torch.arange(C, device=device).repeat_interleave(L + 1)
        attn = (blk_id[None, :] <= blk_id[:, None])[None, None]

        cache = []
        for blk in self.blocks:
            x, kv = blk(x, attn_mask=attn, return_kv=True, rope=rope)
            cache.append(kv)
        return cache

    @torch.no_grad()
    def extend_prefix(self, cache, tokens: torch.Tensor, actions: torch.Tensor, base: int):
        """Append one stream-A block to an existing cache, rope only.

        ``tokens`` is ``(B, 1, L)`` and ``actions`` is ``(B, 1)`` for the single
        frame being committed, whose absolute frame index is ``base``. The new
        block attends the cached prefix and itself, exactly as it would have in
        a full pass, because rope leaves cached keys valid at their original
        positions no matter how far the window has since moved.

        Absolute position embeddings cannot support this: every block's
        embedding is indexed by its slot in the window, so a slide invalidates
        the whole cache. That is the premise this method removes.
        """
        if self.pos_encoding != "rope":
            raise RuntimeError("extend_prefix requires rope positions")
        B, _, L = tokens.shape
        device = tokens.device
        a = torch.cat([self.tok_emb(tokens), self.act_emb(actions).unsqueeze(2)], dim=2)
        a = a + self._type_bias(L + 1, device)
        a = a + self.seg_emb.weight[0]
        x = a.reshape(B, L + 1, -1)
        rope = build_rope(self.a_positions(1, device, base), self.d_head, device=device)

        out = []
        for blk, kv in zip(self.blocks, cache):
            x, new_kv = blk(x, attn_mask=None, prefix_kv=kv, return_kv=True, rope=rope)
            out.append((torch.cat([kv[0], new_kv[0]], dim=2), torch.cat([kv[1], new_kv[1]], dim=2)))
        return out

    @torch.no_grad()
    def decode_logits(self, cache, target: torch.Tensor, frame_index: int):
        """Logits for one target frame given a cached prefix.

        ``target`` is ``(B, L)`` with :attr:`mask_token` in the slots still to
        be filled. No attention mask is needed: the target block attends to the
        whole prefix and to itself.
        """
        B, L = target.shape
        device = target.device
        x = self.tok_emb(target)
        tb = self._type_bias(L, device)
        x = x + (tb[:, 0] if torch.is_tensor(tb) else tb)
        x = x + self.seg_emb.weight[1]
        rope = None
        if self.pos_encoding == "rope":
            # frame_index is the absolute frame number under rope, so the offset
            # between this query and any cached key is the true distance.
            pos = self.a_positions(1, device, frame_index)[:L]
            rope = build_rope(pos, self.d_head, device=device)
        else:
            x = x + self.pos_emb.weight[:L]
            x = x + self.frame_emb.weight[frame_index]
        for blk, kv in zip(self.blocks, cache):
            x = blk(x, attn_mask=None, prefix_kv=kv, rope=rope)
        return self.head(self.ln_f(x))
