"""Cache carrying: the latency-versus-exactness curve.

    python -m ngx.eval.cache_cadence --config configs/small.yaml [--untrained]

Carrying the KV cache across a frame boundary is exact under rope *until the
window slides*. Then the oldest block is evicted, and every retained block still
carries the history it computed while that block was visible -- history a fresh
recompute would not have. That is a property of block-causal attention, not of
the position encoding, and it is out of distribution for a model trained on
fixed windows. So it is measured, not asserted.

Three regimes over the same starts, the same action sequences and greedy
decoding, so any divergence is caused by the approximation and nothing else:

* **full recompute** every frame -- the exact reference, and the slowest.
* **carry, never refreshed** -- the approximation left to accumulate.
* **carry, refreshed every K frames** -- rebuild from scratch every K frames,
  bounding how far the approximation can drift.

Each regime reports per-frame latency and, against the full-recompute rollout,
PSNR and mean tokens differing per frame. Divergence compounds once it starts,
because a different frame becomes different context, so the per-k curve is
reported and not just a mean.

If the untrained flag is used the model is randomly initialised in the config's
shape, and the divergence numbers describe *that* model. Latency does not depend
on weights; exactness might. The table is regenerated on every trained rope
checkpoint, and the default cadence is whatever the trained table supports.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ..config import find_ckpt, load_config, pick_device
from ..envs import make_env
from ..infer.engine import EngineConfig, NeuralGameEngine
from ..infer.load import load_models
from ..models.dynamics import DynamicsTransformer
from ..models.vqvae import VQVAE
from ..train.common import load_ckpt
from .baselines import PSNR_MAX, psnr_capped

MARKS = (1, 8, 16, 32, 60)


def starts_from_env(cfg, num_actions: int, context: int, n: int, prefix_len: int, horizon: int):
    """Matched starts: a replayed prefix for context, then a fixed random plan."""
    out = []
    rng = np.random.default_rng(0)
    for s in range(n):
        prefix = [int(rng.integers(num_actions)) for _ in range(prefix_len)]
        plan = [int(rng.integers(num_actions)) for _ in range(horizon)]
        env = make_env(cfg["data"]["env"], frame_size=64,
                       frame_skip=cfg["data"]["frame_skip"], seed=s, episode_timeout=0)
        try:
            ctx = []
            f = env.reset()
            for i in range(prefix_len - 1):
                ctx.append(f)
                f, _ = env.step(prefix[i])
            ctx.append(f)
        finally:
            env.close()
        out.append((np.asarray(ctx[-context:]),
                    np.asarray(prefix[prefix_len - context : prefix_len - 1] + [0]), plan))
    return out


@torch.no_grad()
def rollout(engine, ctx_f, ctx_a, plan, warmup: int):
    engine.reset(ctx_f, ctx_a)
    frames, toks, times = [], [], []
    for i, a in enumerate(plan):
        t0 = time.perf_counter()
        f = engine.step(a)
        dt = time.perf_counter() - t0
        frames.append(f)
        toks.append(engine.tokens[-1].clone())
        if i >= warmup:
            times.append(dt)
    return np.stack(frames), torch.stack(toks), float(np.mean(times)) * 1000


@torch.no_grad()
def perturbation_vs_depth(dyn, engine, starts, horizon: int):
    """How much eviction perturbs the logits at each depth, *without* steering.

    Runs the exact rollout and, alongside it, a shadow carried cache that is
    extended with the exact rollout's own windows and never allowed to steer.
    At every frame both caches decode the same window, so the comparison
    isolates the approximation from its own compounding. Reports, per eviction
    depth, the first-pass logit perturbation in units of the logit standard
    deviation (how close it came to flipping an argmax) and whether the full
    MaskGIT decode would actually have produced different tokens.
    """
    C, L = engine.C, engine.L
    per_k: dict[int, list] = {}
    for cf, ca, plan in starts:
        engine.reset(cf, ca)
        cache, t0 = None, 0
        for k, a in enumerate(plan[:horizon], start=1):
            engine.actions[-1] = a
            T, A = engine.tokens[None].clone(), engine.actions[None].clone()
            fresh = dyn.encode_prefix(T, A, base=t0)
            if cache is None:
                cache = dyn.encode_prefix(T[:, : C - 1], A[:, : C - 1], base=t0)
            carried = dyn.extend_prefix(cache, T[:, C - 1 :], A[:, C - 1 :], base=t0 + C - 1)
            fi = t0 + C
            blank = torch.full((L,), dyn.mask_token, dtype=torch.long, device=T.device)
            lf = dyn.decode_logits(fresh, blank[None], fi)[0]
            lc = dyn.decode_logits(carried, blank[None], fi)[0]
            rel = float((lf - lc).abs().max() / lf.std().clamp_min(1e-8))
            tf = engine._decode_maskgit(lambda cur: dyn.decode_logits(fresh, cur[None], fi)[0])
            tc = engine._decode_maskgit(lambda cur: dyn.decode_logits(carried, cur[None], fi)[0])
            per_k.setdefault(k, []).append((rel, int((tf != tc).sum())))
            cache = [(kk[:, :, L + 1 :], vv[:, :, L + 1 :]) for kk, vv in carried]
            t0 += 1
            engine.step(a)
    out = {}
    for k, vals in per_k.items():
        v = np.asarray(vals, dtype=float)
        out[k] = {"rel": float(v[:, 0].mean()), "rel_max": float(v[:, 0].max()),
                  "tok": float(v[:, 1].mean()), "any": float((v[:, 1] > 0).mean())}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--starts", type=int, default=6)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--prefix", type=int, default=40)
    p.add_argument("--warmup", type=int, default=3, help="frames excluded from timing")
    p.add_argument("--refresh", type=int, nargs="*", default=[4, 8, 16])
    p.add_argument("--untrained", action="store_true",
                   help="build a randomly initialised rope model in the config's shape")
    p.add_argument("--out", default="docs/CACHE_CADENCE.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    torch.manual_seed(0)

    if a.untrained:
        tc = load_ckpt(find_ckpt(cfg, "tokenizer", "vqvae.pt"), map_location="cpu")
        vq = VQVAE(ch=tc["cfg"]["tokenizer"]["ch"], embed_dim=tc["cfg"]["tokenizer"]["embed_dim"],
                   num_codes=tc["cfg"]["tokenizer"]["num_codes"],
                   n_res=tc["cfg"]["tokenizer"].get("n_res", 2))
        vq.load_state_dict(tc["model"])
        d = cfg["dynamics"]
        dyn = DynamicsTransformer(
            num_codes=vq.num_codes, num_actions=6, tokens_per_frame=vq.tokens_per_frame,
            context=d["context"], d_model=d["d_model"], n_layers=d["n_layers"],
            n_heads=d["n_heads"], pos_encoding="rope",
        )
        weights = "untrained (random init, config shape)"
    else:
        vq, dyn, dck = load_models(cfg, device)
        if getattr(dyn, "pos_encoding", "absolute") != "rope":
            raise SystemExit(
                "checkpoint uses absolute positions and cannot carry a cache; "
                "pass --untrained to measure on a rope model of the same shape"
            )
        weights = f"trained ({find_ckpt(cfg, 'dynamics', 'dynamics.pt')})"
    vq, dyn = vq.to(device).eval(), dyn.to(device).eval()

    ic = cfg["infer"]
    base = dict(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"], temperature=0.0, top_k=0)
    regimes = [("full recompute every frame", dict(carry_cache=False))]
    regimes += [(f"carry, refresh every {k}", dict(carry_cache=True, carry_refresh=k))
                for k in sorted(a.refresh)]
    regimes += [("carry, never refreshed", dict(carry_cache=True, carry_refresh=0))]

    starts = starts_from_env(cfg, dyn.num_actions, dyn.context, a.starts, a.prefix, a.horizon)
    print(f"{weights} | {a.starts} starts x {a.horizon} frames | greedy | device {device}\n")

    ref = {}
    rows = []
    for label, delta in regimes:
        eng = NeuralGameEngine(vq, dyn, EngineConfig(**base, **delta), device=device, memory=None)
        assert eng._carry == delta["carry_cache"], f"{label}: carry flag did not take effect"
        lat, psnr_k, tok_k, ident_k = [], [], [], []
        for s, (cf, ca, plan) in enumerate(starts):
            frames, toks, ms = rollout(eng, cf, ca, plan, a.warmup)
            lat.append(ms)
            if s not in ref:
                ref[s] = (frames, toks)
            rf, rt = ref[s]
            psnr_k.append([psnr_capped(frames[k][None], rf[k][None]) for k in range(a.horizon)])
            tok_k.append([int((toks[k] != rt[k]).sum()) for k in range(a.horizon)])
            ident_k.append([bool(np.array_equal(frames[k], rf[k])) for k in range(a.horizon)])
        psnr_k, tok_k, ident_k = (np.asarray(x) for x in (psnr_k, tok_k, ident_k))
        r = {
            "label": label,
            "ms": float(np.mean(lat)),
            "psnr_mean": float(psnr_k.mean()),
            "tok_mean": float(tok_k.mean()),
            "ident": float(ident_k.mean()),
            "psnr_at": {k: float(psnr_k[:, k - 1].mean()) for k in MARKS if k <= a.horizon},
            "tok_at": {k: float(tok_k[:, k - 1].mean()) for k in MARKS if k <= a.horizon},
            "first_div": [int(np.argmax(~row)) + 1 if (~row).any() else None for row in ident_k],
        }
        rows.append(r)
        exact = r["ident"] == 1.0
        print(f"  {label:30s} {r['ms']:6.2f} ms/frame   "
              + ("identical to reference" if exact else
                 f"PSNR {r['psnr_mean']:5.1f} dB  tokens differ {r['tok_mean']:4.1f}/64  "
                 f"frames identical {100 * r['ident']:5.1f}%"))

    print("\nun-steered perturbation vs eviction depth (reference rollout, shadow carried cache):")
    depth_eng = NeuralGameEngine(vq, dyn, EngineConfig(**base, carry_cache=False), device=device)
    depth = perturbation_vs_depth(dyn, depth_eng, starts, min(a.horizon, 24))
    DEPTHS = [k for k in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24) if k in depth]
    for k in DEPTHS:
        v = depth[k]
        print(f"  depth {k:2d}: max|dlogit| {v['rel']:.3f} sd (worst start {v['rel_max']:.3f})  "
              f"4-pass tokens differ {v['tok']:4.1f}/64 on {100 * v['any']:.0f}% of starts")

    ref_ms = rows[0]["ms"]
    lines = [
        "# Cache carrying: latency versus exactness",
        "",
        f"Weights: {weights}. {a.starts} matched starts (replayed prefix in VizDoom, then a "
        f"fixed random action plan), {a.horizon} frames each, greedy decoding, "
        f"{a.warmup} warmup frames excluded from timing. Device `{device}`.",
        "",
        "Every regime starts from the same context and applies the same actions. The "
        "reference is a full recompute of the prefix cache on every frame, which is exact. "
        "Divergence is measured against that rollout frame by frame; once two rollouts "
        "differ, the difference feeds back through the context, so it compounds and is "
        "reported per k as well as on average.",
        "",
        "| regime | ms/frame | vs. full | mean PSNR vs ref | mean tokens differing | frames identical | first divergence (frame, per start) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        exact = r["ident"] == 1.0
        fd = ", ".join("never" if x is None else str(x) for x in r["first_div"])
        lines.append(
            f"| {r['label']} | {r['ms']:.2f} | {ref_ms / r['ms']:.2f}x | "
            + ("identical" if exact else f"{r['psnr_mean']:.1f} dB")
            + f" | {r['tok_mean']:.1f} / 64 | {100 * r['ident']:.1f}% | {fd} |"
        )
    lines += [
        "",
        "Per-k divergence from the full-recompute rollout, mean over starts. PSNR is clipped "
        f"at {PSNR_MAX:.0f} dB where frames are byte-identical.",
        "",
        "| regime | " + " | ".join(f"k={k}" for k in MARKS if k <= a.horizon) + " |",
        "|---" * (1 + len([k for k in MARKS if k <= a.horizon])) + "|",
    ]
    for r in rows[1:]:
        lines.append(f"| {r['label']} | " + " | ".join(
            f"{r['psnr_at'][k]:.1f} dB / {r['tok_at'][k]:.1f} tok" for k in MARKS if k <= a.horizon
        ) + " |")

    lines += [
        "",
        "## The approximation itself, measured without compounding",
        "",
        "A shadow carried cache is extended along the exact rollout's own windows and never "
        "allowed to steer, so at every frame it decodes the same window the exact cache does. "
        "`max dlogit` is the first-pass logit perturbation in units of the logit standard "
        "deviation (how close it came to flipping a decision); `tokens differ` is whether the "
        "full MaskGIT decode actually came out different on that frame.",
        "",
        "| eviction depth | max dlogit (sd), mean over starts | worst start | 4-pass tokens differ | starts affected |",
        "|---|---|---|---|---|",
    ]
    for k in DEPTHS:
        v = depth[k]
        lines.append(f"| {k} | {v['rel']:.3f} | {v['rel_max']:.3f} | {v['tok']:.1f} / 64 | "
                     f"{100 * v['any']:.0f}% |")
    lines += [
        "",
        "Depth 1 is exact by construction. The perturbation grows with depth for the first "
        "several frames as each retained block inherits history a fresh recompute would not "
        "give it, and a refresh every K frames caps the depth at K-1. Once a single token "
        "differs, the rollouts have different context and nothing recovers them, so a refresh "
        "bounds the *size* of the approximation and not the divergence of the rollout.",
        "",
    ]

    never = rows[-1]
    if never["ident"] == 1.0:
        verdict = (
            f"**Carry it, never refresh.** Over {a.starts * a.horizon} frames the never-refreshed "
            f"rollout is byte-identical to full recompute at {ref_ms / never['ms']:.2f}x the "
            "throughput. The eviction approximation exists in principle and does not change a "
            "single output here."
        )
    else:
        best = next((r for r in rows[1:-1] if r["ident"] == 1.0), None)
        if best is not None:
            verdict = (
                f"**Refresh cadence is the answer.** Never refreshing diverges "
                f"({100 * (1 - never['ident']):.1f}% of frames differ, mean {never['tok_mean']:.1f} "
                f"tokens); `{best['label']}` is byte-identical to full recompute at "
                f"{ref_ms / best['ms']:.2f}x. That is the default."
            )
        else:
            verdict = (
                "**No carrying regime is exact on this model.** The table gives the trade; the "
                f"never-refreshed regime differs on {100 * (1 - never['ident']):.1f}% of frames "
                f"by {never['tok_mean']:.1f} tokens on average for {ref_ms / never['ms']:.2f}x. "
                "Whether that is acceptable is a judgement about the demo, and it is stated "
                "here as a number rather than made silently."
            )
    lines += ["", verdict, ""]
    if a.untrained:
        lines += [
            "**These divergence numbers describe an untrained model, and untrained models are "
            "not a usable proxy in either direction.** Two random initialisations of this exact "
            "shape were checked: one never diverged in twelve frames, the other diverged at "
            "frame 5 on every start. Sensitivity to a fixed-size perturbation depends on how "
            "close the model's decisions sit to a tie, and random init puts that anywhere. "
            "Latency does not depend on weights and stands. Exactness is decided on the first "
            "trained rope checkpoint, and until then the engine defaults to the exact path.",
            "",
        ]
    lines += ["Regenerate with `python -m ngx.eval.cache_cadence --config configs/small.yaml"
              + (" --untrained" if a.untrained else "") + "`.", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
