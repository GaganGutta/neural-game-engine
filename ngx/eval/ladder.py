"""Ladder report: every rung against every pre-registered rule, from checkpoints.

    python -m ngx.eval.ladder --config configs/ladder/2m.yaml \
        --runs ladder-2m-s0 ladder-2m-s1 ladder-2m-s2 --group "2M, seeds 0-2"

Loads ``runs/<name>/dynamics/dynamics.pt`` for each run (the config's ``name``
is overridden per run, everything else is shared) and computes the columns
``docs/LADDER_PREREG.md`` asks for:

* params, context, tokens seen, epochs, held-out loss and cold accuracy (all
  read from the checkpoint the trainer wrote);
* one-step PSNR on **moving** transitions, greedy, with copy-last-frame and the
  tokenizer ceiling on the same windows, and the percent of headroom captured;
* closed-loop lead over a frozen frame at k = 1, 8, 16, 32;
* return-to-place consistency (model vs. the game's own ceiling) and drift at
  k = 100 and 1000, from the same reference trajectory ``drift.py`` uses;
* the bucketed hold-still table (k = 2-10 identical rate and tokens changed
  when it moves; k > 10 identical rate) against the real game.

Results accumulate in ``docs/ladder_results.json`` keyed by run name, so later
rungs are added without recomputing earlier ones, and ``docs/LADDER.md`` is
regenerated from that file every time. For any group with more than one run
the max-minus-min spread of one-step PSNR is printed: per the pre-registration
that spread is the ladder's resolution, and rung-to-rung differences inside it
are read as no effect.

Every evaluation here runs on the same held-out windows, the same reference
trajectory, the same revisit pairs and the same matched starts for every run,
so the comparison across rungs is paired rather than merely repeated.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..config import load_config, pick_device
from ..infer.engine import EngineConfig, NeuralGameEngine
from ..infer.load import load_models
from .baselines import agg, closed_loop, one_step, sample_windows, still_compare
from .drift import evaluate as drift_evaluate
from .drift import find_revisits, reference_trajectory

RESULTS = "docs/ladder_results.json"
CL_MARKS = (1, 8, 16, 32)


def evaluate_run(cfg: dict, name: str, device, shared: dict, a) -> dict:
    run_cfg = dict(cfg)
    run_cfg["name"] = name
    vq, dyn, ck = load_models(run_cfg, device)
    ic = cfg["infer"]
    engine = NeuralGameEngine(
        vq, dyn,
        EngineConfig(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"],
                     temperature=0.0, top_k=0, carry_cache=False),
        device=device, memory=None,
    )
    C = dyn.context
    params = sum(p.numel() for p in dyn.parameters())

    # windows are per-context, so cache them by C
    if C not in shared["one_step"]:
        shared["one_step"][C] = sample_windows(cfg["data"]["root"], C + 1, a.windows, seed=0)
        shared["closed"][C] = sample_windows(cfg["data"]["root"], C + a.horizon, a.rollouts, seed=1)
    if "traj" not in shared:
        frames, actions, poses, used_seed = reference_trajectory(cfg, a.drift_steps + 64, 0)
        shared["traj"] = (frames, actions, poses, used_seed)
    frames, actions, poses, used_seed = shared["traj"]
    pairs_key = ("pairs", C)
    if pairs_key not in shared:
        shared[pairs_key] = find_revisits(poses, start=C, min_gap=60, pos_tol=40, ang_tol=20)

    # -- one-step, moving subset only --------------------------------------
    recs = one_step({"greedy": engine}, vq, shared["one_step"][C], device)
    moving = [r for r in recs if not r["static"]]
    m_mean, m_med = agg([r["greedy|psnr"] for r in moving])
    c_mean, _ = agg([r["copy_psnr"] for r in moving])
    ceil_mean, _ = agg([r["ceiling_psnr"] for r in moving])
    head = ceil_mean - c_mean
    acc = float(np.mean([r["greedy|acc"] for r in moving]))

    # -- closed loop --------------------------------------------------------
    model, frozen, _ceil = closed_loop(engine, vq, shared["closed"][C], device, a.horizon)
    lead = {k: float(model[:, k - 1].mean() - frozen[:, k - 1].mean())
            for k in CL_MARKS if k <= a.horizon}

    # -- drift and return-to-place -----------------------------------------
    torch.manual_seed(0)
    dr = drift_evaluate(engine, frames, actions, poses, a.drift_steps, shared[pairs_key])

    # -- hold-still buckets --------------------------------------------------
    names = list(ck.get("action_names") or [])
    noop = names.index("-") if "-" in names else 0
    real_still, model_still, _exc = still_compare(cfg, engine, vq, device, 8, 60, 40, noop)

    return {
        "name": name,
        "params": int(params),
        "context": int(C),
        "d_model": int(dyn.d_model),
        "n_layers": int(dyn.n_layers),
        "seed": ck.get("seed"),
        "tokens_seen": ck.get("tokens_seen"),
        "epochs": ck.get("epochs"),
        "val_loss": ck.get("val_loss"),
        "cold_acc": ck.get("cold_acc"),
        "one_step_moving": m_mean,
        "one_step_moving_median": m_med,
        "one_step_token_acc": acc,
        "copy_moving": c_mean,
        "ceiling_moving": ceil_mean,
        "headroom_pct": 100.0 * (m_mean - c_mean) / max(head, 1e-9),
        "n_moving": len(moving),
        "lead": {str(k): v for k, v in lead.items()},
        "revisit_model": dr["model_revisit_psnr"],
        "revisit_game": dr["env_revisit_psnr"],
        "revisit_pairs": dr["revisits"],
        "drift_k100": dr["curve"].get(100),
        "drift_k1000": dr["curve"].get(1000),
        "still_k2_10_ident": model_still["k=2-10"]["ident"],
        "still_k2_10_churn": model_still["k=2-10"]["churn_when_moved"],
        "still_k10_ident": model_still["k>10"]["ident"],
        "still_ref_k2_10_ident": real_still["k=2-10"]["ident"],
        "still_ref_k2_10_churn": real_still["k=2-10"]["churn_when_moved"],
        "traj_seed": used_seed,
    }


def render(results: dict, groups: dict) -> str:
    L = ["# Scaling ladder", "",
         "Every rung is scored on the same held-out windows, reference trajectory, revisit "
         "pairs and matched starts, so rows are paired comparisons. Read every difference "
         "against the seed spread at the bottom, which the pre-registration defines as the "
         "ladder's resolution. Rules and their branches are in "
         "[LADDER_PREREG.md](LADDER_PREREG.md).", "",
         "| run | params | ctx | tokens seen | epochs | val loss | cold acc | one-step moving | "
         "copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place "
         "(game) | still k=2-10 ident / churn | still k>10 ident |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        ld = r["lead"]
        L.append(
            f"| {r['name']} | {r['params'] / 1e6:.1f}M | {r['context']} | "
            f"{(r['tokens_seen'] or 0) / 1e9:.2f}B | {(r['epochs'] or 0):.2f} | "
            f"{(r['val_loss'] or float('nan')):.3f} | {(r['cold_acc'] or float('nan')):.3f} | "
            f"{r['one_step_moving']:.2f} dB | {r['copy_moving']:.2f} | {r['ceiling_moving']:.2f} | "
            f"**{r['headroom_pct']:.0f}%** | "
            + " | ".join(f"{ld.get(str(k), float('nan')):+.2f}" for k in CL_MARKS)
            + f" | {r['revisit_model']:.2f} ({r['revisit_game']:.2f}) dB | "
            f"{100 * r['still_k2_10_ident']:.0f}% / {r['still_k2_10_churn']:.1f} | "
            f"{100 * r['still_k10_ident']:.1f}% |"
        )
    any_r = next(iter(results.values()), None)
    if any_r:
        L += ["", f"Real-game hold-still reference: k=2-10 {100 * any_r['still_ref_k2_10_ident']:.1f}% "
              f"identical, {any_r['still_ref_k2_10_churn']:.1f} tokens when it moves; k>10 100%. "
              f"Return-to-place is shown as model (game ceiling). Headroom is on the moving "
              f"subset: (model - copy) / (ceiling - copy). Lead is model minus frozen-frame PSNR."]
    L += ["", "## Seed spread, the ladder's resolution", ""]
    for g, names in groups.items():
        rs = [results[n] for n in names if n in results]
        if len(rs) < 2:
            continue
        v = [r["one_step_moving"] for r in rs]
        h = [r["headroom_pct"] for r in rs]
        L.append(f"- **{g}** ({len(rs)} runs): one-step moving PSNR "
                 f"{min(v):.2f} to {max(v):.2f} dB, **spread {max(v) - min(v):.2f} dB**; "
                 f"headroom {min(h):.0f}% to {max(h):.0f}%, spread {max(h) - min(h):.0f} points; "
                 f"val loss {min(r['val_loss'] for r in rs):.3f} to "
                 f"{max(r['val_loss'] for r in rs):.3f}.")
    L += ["", "Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> "
          "--group <label>`; results accumulate in `docs/ladder_results.json`.", ""]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    p.add_argument("--group", default=None, help="label for the seed-spread line")
    p.add_argument("--device", default="auto")
    p.add_argument("--windows", type=int, default=800)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--drift-steps", type=int, default=1000)
    p.add_argument("--out", default="docs/LADDER.md")
    p.add_argument("--results", default=RESULTS, help="json store; results accumulate here")
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    store = (json.load(open(a.results)) if os.path.exists(a.results)
             else {"results": {}, "groups": {}})
    shared: dict = {"one_step": {}, "closed": {}}

    for name in a.runs:
        if name in store["results"] and not a.recompute:
            print(f"{name}: cached")
            continue
        print(f"{name}: evaluating ...", flush=True)
        r = evaluate_run(cfg, name, device, shared, a)
        store["results"][name] = r
        print(f"  one-step moving {r['one_step_moving']:.2f} dB (copy {r['copy_moving']:.2f}, "
              f"ceiling {r['ceiling_moving']:.2f}) -> headroom {r['headroom_pct']:.0f}% | "
              f"val {r['val_loss']:.3f} | tokens {(r['tokens_seen'] or 0) / 1e9:.2f}B "
              f"epochs {(r['epochs'] or 0):.2f} | lead " +
              " ".join(f"k{k}:{r['lead'][str(k)]:+.2f}" for k in CL_MARKS) +
              f" | r2p {r['revisit_model']:.2f}/{r['revisit_game']:.2f} | "
              f"still2-10 {100 * r['still_k2_10_ident']:.0f}%/{r['still_k2_10_churn']:.1f} "
              f"k>10 {100 * r['still_k10_ident']:.1f}%")
    if a.group:
        store["groups"][a.group] = list(a.runs)

    os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    with open(a.results, "w") as f:
        json.dump(store, f, indent=1)
    with open(a.out, "w") as f:
        f.write(render(store["results"], store["groups"]))
    print(f"\nwrote {a.out} and {a.results}")


if __name__ == "__main__":
    main()
