"""The thing you actually play.

Holds a rolling window of tokenized frames and the actions between them, and
turns "player pressed W" into the next frame. No game engine is involved.

Where the frame time goes, and what each switch does about it:

* The context prefix is identical across every decoding iteration of a frame,
  so its keys and values are computed once per frame instead of once per
  iteration (``use_cache``). Saves a factor of roughly the iteration count.
* Raster decoding needs one forward pass per token, so 64 per frame. MaskGIT
  fills every remaining slot each pass and keeps the confident ones, so 8
  passes cover the frame (``decode``).
* bf16/fp16 autocast, ``torch.compile`` and int8 dynamic quantisation each
  attack the cost of a single pass (``dtype``, ``compile``, ``int8``).

Retrieval memory, when enabled, swaps the oldest context slots for frames
recorded the last time the player stood somewhere similar. That keeps the block
count at exactly what the model was trained on, so no retraining is needed.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..models.dynamics import DynamicsTransformer
from ..models.vqvae import VQVAE


@dataclass
class EngineConfig:
    decode: str = "maskgit"        # 'maskgit' | 'raster'
    maskgit_steps: int = 8
    temperature: float = 1.0
    top_k: int = 50
    use_cache: bool = True
    dtype: str = "fp32"            # 'fp32' | 'bf16' | 'fp16'
    compile: bool = False
    int8: bool = False


def _autocast(device: torch.device, dtype: str):
    if dtype == "fp32":
        return contextlib.nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=torch_dtype)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if not k or k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


class NeuralGameEngine:
    def __init__(
        self,
        vqvae: VQVAE,
        dynamics: DynamicsTransformer,
        cfg: EngineConfig | None = None,
        device: torch.device | str = "cpu",
        memory=None,
    ) -> None:
        self.device = torch.device(device)
        self.cfg = cfg or EngineConfig()
        self.vq = vqvae.to(self.device).eval()
        self.model = dynamics.to(self.device).eval()
        self.memory = memory

        if self.cfg.int8:
            from .quantize import quantize_int8

            self.model = quantize_int8(self.model)  # returns a new module

        # Bind the entry points on the engine rather than on the module. Two
        # reasons: torch.compile(module) only wraps forward(), which the cached
        # path never calls -- compiling the module would measure wrapper
        # overhead and report it as a speedup -- and the benchmark hands the
        # same model to every row, so mutating it would leak one row's
        # optimisation into the next.
        self._encode_prefix = self.model.encode_prefix
        self._decode_logits = self.model.decode_logits
        self._forward = self.model.__call__
        if self.cfg.compile:
            self._encode_prefix = torch.compile(self._encode_prefix, dynamic=False)
            self._decode_logits = torch.compile(self._decode_logits, dynamic=False)
            self._forward = torch.compile(self._forward, dynamic=False)

        self.C = self.model.context
        self.L = self.model.L
        self.tokens = torch.zeros(self.C, self.L, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(self.C, dtype=torch.long, device=self.device)
        self._frame = np.zeros((64, 64, 3), np.uint8)
        self.last_retrieved = 0

    # -- lifecycle ----------------------------------------------------------
    @torch.no_grad()
    def reset(self, frames: np.ndarray, actions: np.ndarray | None = None) -> np.ndarray:
        """Seed the context window with real frames.

        ``frames`` is ``(C, H, W, 3)`` uint8 -- the handful of real frames the
        model needs before it can take over. ``actions`` is ``(C,)``; the last
        entry is overwritten by the first :meth:`step`.
        """
        if len(frames) < self.C:
            raise ValueError(f"need {self.C} seed frames, got {len(frames)}")
        x = torch.from_numpy(np.asarray(frames[-self.C :], dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(0, 3, 1, 2).to(self.device)
        self.tokens = self.vq.encode_indices(x).long()
        self.actions = (
            torch.zeros(self.C, dtype=torch.long, device=self.device)
            if actions is None
            # np.array copies: seeds often come straight off a read-only memmap.
            else torch.as_tensor(
                np.array(actions[-self.C :], dtype=np.int64), device=self.device
            )
        )
        if self.memory is not None:
            self.memory.reset()
        self._frame = np.asarray(frames[-1], dtype=np.uint8)
        return self._frame

    @torch.no_grad()
    def step(self, action: int) -> np.ndarray:
        """Advance the world by one frame under ``action``."""
        # The action the player just pressed belongs to the newest context
        # frame: it is what turns that frame into the one about to be drawn.
        self.actions[-1] = int(action)

        tokens, actions = self.tokens, self.actions
        self.last_retrieved = 0
        if self.memory is not None:
            tokens, actions, self.last_retrieved = self.memory.augment(tokens, actions)

        with _autocast(self.device, self.cfg.dtype):
            new_tokens = self._predict_frame(tokens, actions)

        if self.memory is not None:
            self.memory.write(self.tokens[-1], int(action))

        # Slide the window: the predicted frame becomes the newest context.
        self.tokens = torch.roll(self.tokens, -1, dims=0)
        self.tokens[-1] = new_tokens
        self.actions = torch.roll(self.actions, -1, dims=0)
        self.actions[-1] = 0

        px = self.vq.decode_indices(new_tokens[None])[0]
        self._frame = (
            ((px.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        )
        return self._frame

    @property
    def frame(self) -> np.ndarray:
        return self._frame

    # -- decoding -----------------------------------------------------------
    def _logits_fn(self, tokens: torch.Tensor, actions: torch.Tensor):
        """Return ``f(target_tokens) -> logits``, cached or not.

        The uncached variant is kept because it is the honest "before" number
        in the benchmark table: it recomputes the whole prefix on every
        decoding iteration, which is what a naive implementation does.
        """
        prefix_tok, prefix_act = tokens[None], actions[None]
        if self.cfg.use_cache:
            cache = self._encode_prefix(prefix_tok, prefix_act)
            return lambda cur: self._decode_logits(cache, cur[None], self.C)[0]

        def uncached(cur: torch.Tensor) -> torch.Tensor:
            full = torch.cat([prefix_tok, cur[None, None]], dim=1)
            act = torch.cat([prefix_act, torch.zeros_like(prefix_act[:, :1])], dim=1)
            mask = torch.zeros(1, self.C, self.L, dtype=torch.bool, device=self.device)
            mask[:, -1] = cur == self.model.mask_token
            return self._forward(full, act, mask)[0, -1]

        return uncached

    def _sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a token per position; also return the sampled token's prob."""
        logits = logits.float()
        if self.cfg.temperature <= 0:
            probs = torch.softmax(logits, -1)
            tok = logits.argmax(-1)
        else:
            filtered = _top_k_filter(logits, self.cfg.top_k) / self.cfg.temperature
            probs = torch.softmax(filtered, -1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        return tok, probs.gather(-1, tok[:, None]).squeeze(-1)

    def _predict_frame(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits_fn = self._logits_fn(tokens, actions)
        if self.cfg.decode == "raster":
            return self._decode_raster(logits_fn)
        return self._decode_maskgit(logits_fn)

    def _decode_raster(self, logits_fn) -> torch.Tensor:
        """One forward pass per token, left to right. The AR baseline."""
        cur = torch.full((self.L,), self.model.mask_token, dtype=torch.long, device=self.device)
        for j in range(self.L):
            tok, _ = self._sample(logits_fn(cur)[j : j + 1])
            cur[j] = tok[0]
        return cur

    def _decode_maskgit(self, logits_fn) -> torch.Tensor:
        """Fill every masked slot each pass, keep the most confident ones.

        The number left masked after pass ``i`` follows a cosine schedule from
        ``L`` down to 0, so early passes commit only the tokens the model is
        sure about and later passes fill in the rest given that scaffolding.
        """
        K = max(int(self.cfg.maskgit_steps), 1)
        mask_id = self.model.mask_token
        cur = torch.full((self.L,), mask_id, dtype=torch.long, device=self.device)
        for i in range(K):
            tok, conf = self._sample(logits_fn(cur))
            known = cur != mask_id
            tok = torch.where(known, cur, tok)
            conf = torch.where(known, torch.full_like(conf, float("inf")), conf)

            n_masked = int(math.floor(self.L * math.cos(math.pi / 2 * (i + 1) / K)))
            if n_masked <= 0 or i == K - 1:
                return tok
            keep = conf.argsort(descending=True)[: self.L - n_masked]
            cur = torch.full_like(cur, mask_id)
            cur[keep] = tok[keep]
        return cur
