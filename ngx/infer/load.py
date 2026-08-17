"""Rebuild a trained engine from a run directory.

Shared by ``play.py``, the benchmark harness and the drift evaluation so all
three are demonstrably running the same thing.
"""

from __future__ import annotations

import os

import torch

from ..config import pick_device, run_dir
from ..models.dynamics import DynamicsTransformer
from ..models.vqvae import VQVAE
from ..train.common import load_ckpt
from .engine import EngineConfig, NeuralGameEngine
from .memory import RetrievalMemory


#: repo root, so shipped checkpoints resolve regardless of the working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_ckpt(cfg: dict, stage: str, filename: str) -> str:
    """Prefer a locally trained checkpoint, fall back to the one in the repo.

    Training writes to ``runs/<name>/<stage>/``. A fresh clone has no ``runs/``,
    so ``checkpoints/<name>/`` carries the weights that make ``python play.py``
    work without training anything first.
    """
    local = os.path.join(run_dir(cfg, stage), filename)
    if os.path.exists(local):
        return local
    shipped = os.path.join(_ROOT, "checkpoints", cfg.get("name", "default"), filename)
    if os.path.exists(shipped):
        return shipped
    raise FileNotFoundError(
        f"no checkpoint at {local} or {shipped}. Train it first -- see the README."
    )


def load_models(cfg: dict, device: torch.device):
    vq_path = find_ckpt(cfg, "tokenizer", "vqvae.pt")
    dyn_path = find_ckpt(cfg, "dynamics", "dynamics.pt")

    vck = load_ckpt(vq_path, map_location="cpu")
    tc = vck["cfg"]["tokenizer"]
    vq = VQVAE(ch=tc["ch"], embed_dim=tc["embed_dim"], num_codes=tc["num_codes"],
               n_res=tc.get("n_res", 2))
    vq.load_state_dict(vck["model"])

    dck = load_ckpt(dyn_path, map_location="cpu")
    dc = dck["cfg"]["dynamics"]
    num_actions = dck["model"]["act_emb.weight"].shape[0]
    dyn = DynamicsTransformer(
        num_codes=tc["num_codes"], num_actions=num_actions,
        tokens_per_frame=vq.tokens_per_frame, context=dc["context"],
        d_model=dc["d_model"], n_layers=dc["n_layers"], n_heads=dc["n_heads"],
    )
    dyn.load_state_dict(dck["model"])
    return vq.to(device).eval(), dyn.to(device).eval(), dck


def build_memory(cfg: dict, vq: VQVAE, dyn: DynamicsTransformer, device) -> RetrievalMemory | None:
    mc = cfg.get("memory", {})
    if not mc.get("enabled", False):
        return None
    return RetrievalMemory(
        code_embed=vq.quantizer.embed, tokens_per_frame=dyn.L,
        capacity=mc.get("capacity", 4096), k=mc.get("k", 2),
        min_sim=mc.get("min_sim", 0.9), write_every=mc.get("write_every", 4),
        exclude_recent=mc.get("exclude_recent", 64), device=device,
    )


def load_engine(
    cfg: dict,
    device: str | torch.device = "auto",
    memory: bool | None = None,
    **engine_overrides,
) -> NeuralGameEngine:
    """Build a ready-to-play engine.

    ``engine_overrides`` are :class:`EngineConfig` fields; anything not given
    falls back to the config's ``infer`` block.
    """
    device = pick_device(device) if isinstance(device, str) else device
    vq, dyn, ck = load_models(cfg, device)

    ic = dict(cfg.get("infer", {}))
    fields = EngineConfig().__dict__
    ecfg = EngineConfig(**{k: v for k, v in {**ic, **engine_overrides}.items() if k in fields})

    use_mem = cfg.get("memory", {}).get("enabled", False) if memory is None else memory
    mem = build_memory({**cfg, "memory": {**cfg.get("memory", {}), "enabled": use_mem}},
                       vq, dyn, device)
    engine = NeuralGameEngine(vq, dyn, ecfg, device=device, memory=mem)
    # Carried in the checkpoint so play.py can label the controls without the
    # training data being present.
    engine.action_names = tuple(
        ck.get("action_names") or [str(i) for i in range(dyn.num_actions)]
    )
    return engine
