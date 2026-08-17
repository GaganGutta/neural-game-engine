"""VQ-VAE that turns a 64x64 frame into an 8x8 grid of discrete tokens.

64 tokens per frame is what makes the dynamics model tractable: a 16-frame
context is ~1k tokens, which a small transformer handles at interactive rates.

Codebook is updated by EMA rather than by a codebook loss -- it converges
faster and is less sensitive to the optimizer's learning rate. Dead codes are
restarted from live encoder outputs, without which utilisation on this data
collapses to a few dozen entries and reconstructions get blotchy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.GroupNorm:
    # GroupNorm, not BatchNorm: at play time the batch is 1.
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _norm(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
            _norm(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    """64x64x3 -> 8x8xembed_dim (three stride-2 stages)."""

    def __init__(self, ch: int = 128, embed_dim: int = 64, n_res: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch // 2, 4, 2, 1), nn.SiLU(),      # 32
            nn.Conv2d(ch // 2, ch, 4, 2, 1), nn.SiLU(),     # 16
            nn.Conv2d(ch, ch, 4, 2, 1),                     # 8
            *[ResBlock(ch) for _ in range(n_res)],
            _norm(ch), nn.SiLU(), nn.Conv2d(ch, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """8x8xembed_dim -> 64x64x3."""

    def __init__(self, ch: int = 128, embed_dim: int = 64, n_res: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, ch, 3, padding=1),
            *[ResBlock(ch) for _ in range(n_res)],
            _norm(ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.SiLU(),         # 16
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.SiLU(),    # 32
            nn.ConvTranspose2d(ch // 2, 3, 4, 2, 1),                # 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int = 512,
        dim: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_after: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_codes, self.dim = num_codes, dim
        self.decay, self.eps = decay, eps
        self.restart_after = restart_after
        embed = torch.randn(num_codes, dim) * 0.5
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(num_codes))

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, H, W)`` indices -> ``(B, D, H, W)`` embeddings."""
        z = F.embedding(idx, self.embed)
        return z.permute(0, 3, 1, 2).contiguous()

    def encode(self, z_e: torch.Tensor) -> torch.Tensor:
        """``(B, D, H, W)`` -> ``(B, H, W)`` nearest-code indices."""
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        # ||a - b||^2 expanded; the ||a||^2 term is constant per row and does
        # not affect the argmin, but keeping it makes distances usable.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        return dist.argmin(1).view(b, h, w)

    def forward(self, z_e: torch.Tensor):
        idx = self.encode(z_e)
        z_q = self.lookup(idx)

        if self.training:
            self._ema_update(z_e, idx)

        commit = F.mse_loss(z_e, z_q.detach())
        # Straight-through: gradients flow to the encoder as if quantisation
        # were the identity.
        z_q = z_e + (z_q - z_e).detach()

        with torch.no_grad():
            counts = torch.bincount(idx.reshape(-1), minlength=self.num_codes).float()
            p = counts / counts.sum().clamp_min(1)
            perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
            used = int((counts > 0).sum())
        return z_q, idx, commit, {"perplexity": perplexity.item(), "codes_used": used}

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, idx: torch.Tensor) -> None:
        d = z_e.shape[1]
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        onehot = F.one_hot(idx.reshape(-1), self.num_codes).type(flat.dtype)

        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        # Laplace smoothing keeps rarely-hit codes from exploding when divided.
        cluster = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.embed.copy_(self.embed_avg / cluster.unsqueeze(1))

        # Restart codes nothing maps to, drawing replacements from the current
        # batch of encoder outputs so they land where the data actually is.
        dead = self.cluster_size < self.restart_after
        n_dead = int(dead.sum())
        if n_dead:
            pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            self.embed[dead] = flat[pick]
            self.embed_avg[dead] = flat[pick]
            self.cluster_size[dead] = 1.0


class VQVAE(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        embed_dim: int = 64,
        num_codes: int = 512,
        n_res: int = 2,
        commit_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(ch, embed_dim, n_res)
        self.decoder = Decoder(ch, embed_dim, n_res)
        self.quantizer = VectorQuantizerEMA(num_codes, embed_dim)
        self.commit_beta = commit_beta
        self.num_codes = num_codes
        self.tokens_per_frame = 64  # 8x8

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, idx, commit, stats = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        recon = F.mse_loss(x_hat, x)
        loss = recon + self.commit_beta * commit
        stats = {"recon": recon.item(), "commit": commit.item(), **stats}
        return x_hat, loss, stats

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Frames ``(B, 3, S, S)`` in [-1, 1] -> tokens ``(B, 64)`` int64."""
        idx = self.quantizer.encode(self.encoder(x))
        return idx.flatten(1)

    @torch.no_grad()
    def decode_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tokens ``(B, 64)`` -> frames ``(B, 3, S, S)`` in [-1, 1]."""
        b, n = tokens.shape
        s = int(n**0.5)
        z_q = self.quantizer.lookup(tokens.view(b, s, s))
        return self.decoder(z_q)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """PSNR in dB between two [-1, 1] images."""
    mse = F.mse_loss(x.clamp(-1, 1), y.clamp(-1, 1)).item()
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()
