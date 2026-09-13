"""Ladder report: every rung against every pre-registered rule, from checkpoints.

    python -m ngx.eval.ladder --config configs/ladder/2m.yaml \\
        --runs ladder-2m-t4-s0 ladder-2m-t4-s1 ladder-2m-t4-s2 --group "2M at T*, seeds 0-2"

Loads ``runs/<name>/dynamics/final.pt`` for each run (the config's ``name`` is
overridden per run, everything else is shared). The final checkpoint, not the
best-by-validation ``dynamics.pt``, so every rung is scored at exactly its
token budget. Columns are the ones ``docs/LADDER_PREREG.md`` asks for:

* params, context, tokens seen and epochs (from the checkpoint);
* held-out loss and cold accuracy on fixed validation frames
  (:mod:`ngx.eval.heldout`), identical for every rung and independent of the
  batch size the run trained with;
* one-step PSNR on **moving** transitions, greedy, with copy-last-frame and the
  tokenizer ceiling on the same windows, and the percent of headroom captured;
* closed-loop lead over a frozen frame at k = 1, 8, 16, 32;
* return-to-place consistency (model vs. the game's own ceiling) and drift at
  k = 100 and 1000, from the same reference trajectory ``drift.py`` uses;
* the bucketed hold-still table (k = 2-10 identical rate and tokens changed
  when it moves; k > 10 identical rate) against the real game.

Every rung predicts the same frames, whatever its context length. Windows are
drawn once with ``MAX_CONTEXT`` frames of history and each rung is handed its
own last C of them; revisit pairs are found from frame ``MAX_CONTEXT`` and
every drift rollout starts there. The hold-still starts share a 40-frame
prefix already. So rows are paired comparisons on both ladder axes.

The 8M learning-rate probe is recorded with ``--probe <run>=<lr> ...``: each
run's ``heldout.json`` (written on the training machine by
:mod:`ngx.eval.heldout`, the numbers the winner was picked on) goes into the
same store and is rendered as its own table.

Results accumulate in ``docs/ladder_results.json`` keyed by run name, and
``docs/LADDER.md`` is regenerated from that file every time. A stored row from
an older eval protocol is dropped until it is rescored, never mixed in. For
any group with more than one run the max-minus-min spread of one-step PSNR is
printed: per the pre-registration that spread is the ladder's resolution, and
rung-to-rung differences inside it are read as no effect.
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
from .heldout import MAX_CONTEXT, target_frames
from .heldout import evaluate as heldout_evaluate

RESULTS = "docs/ladder_results.json"
CL_MARKS = (1, 8, 16, 32)
#: bump when a change makes stored rows incomparable with new ones
PROTOCOL = 2


def align(windows, C: int):
    """Hand a C-frame model the tail of windows drawn with ``MAX_CONTEXT``
    frames of history, so its targets are the frames every other rung predicts."""
    off = MAX_CONTEXT - C
    if off < 0:
        raise ValueError(f"context {C} exceeds MAX_CONTEXT {MAX_CONTEXT}")
    return [(f[off:], a[off:]) for f, a in windows]


def shift_pairs(pairs, off: int):
    """Re-index revisit pairs for a trajectory sliced ``off`` frames in."""
    return [(i - off, j - off) for i, j in pairs]


def evaluate_run(cfg: dict, name: str, device, shared: dict, a) -> dict:
    run_cfg = dict(cfg)
    run_cfg["name"] = name
    run_cfg["dynamics_ckpt"] = os.path.join(cfg.get("run_root", "runs"), name, "dynamics", "final.pt")
    vq, dyn, ck = load_models(run_cfg, device)
    ic = cfg["infer"]
    engine = NeuralGameEngine(
        vq, dyn,
        EngineConfig(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"],
                     temperature=0.0, top_k=0, carry_cache=False),
        device=device, memory=None,
    )
    C = dyn.context
    off = MAX_CONTEXT - C
    params = sum(p.numel() for p in dyn.parameters())
    root = cfg["data"]["root"]

    if "one_step" not in shared:
        shared["one_step"] = sample_windows(root, MAX_CONTEXT + 1, a.windows, seed=0)
        shared["closed"] = sample_windows(root, MAX_CONTEXT + a.horizon, a.rollouts, seed=1)
        shared["targets"] = target_frames(root, a.heldout_targets)
        frames, actions, poses, used_seed = reference_trajectory(cfg, a.drift_steps + 64, 0)
        shared["traj"] = (frames, actions, poses, used_seed)
        shared["pairs"] = find_revisits(poses, start=MAX_CONTEXT, min_gap=60, pos_tol=40, ang_tol=20)
    frames, actions, poses, used_seed = shared["traj"]

    # -- held-out loss on fixed frames --------------------------------------
    ho = heldout_evaluate(dyn, root, shared["targets"], device)

    # -- one-step, moving subset only --------------------------------------
    recs = one_step({"greedy": engine}, vq, align(shared["one_step"], C), device)
    moving = [r for r in recs if not r["static"]]
    m_mean, m_med = agg([r["greedy|psnr"] for r in moving])
    c_mean, _ = agg([r["copy_psnr"] for r in moving])
    ceil_mean, _ = agg([r["ceiling_psnr"] for r in moving])
    head = ceil_mean - c_mean
    acc = float(np.mean([r["greedy|acc"] for r in moving]))

    # -- closed loop --------------------------------------------------------
    model, frozen, _ceil = closed_loop(engine, vq, align(shared["closed"], C), device, a.horizon)
    lead = {k: float(model[:, k - 1].mean() - frozen[:, k - 1].mean())
            for k in CL_MARKS if k <= a.horizon}

    # -- drift and return-to-place, every rollout starting at MAX_CONTEXT ----
    torch.manual_seed(0)
    dr = drift_evaluate(engine, frames[off:], actions[off:], poses[off:], a.drift_steps,
                        shift_pairs(shared["pairs"], off))

    # -- hold-still buckets --------------------------------------------------
    names = list(ck.get("action_names") or [])
    noop = names.index("-") if "-" in names else 0
    real_still, model_still, _exc = still_compare(cfg, engine, vq, device, 8, 60, 40, noop)

    return {
        "name": name,
        "protocol": PROTOCOL,
        "ckpt": run_cfg["dynamics_ckpt"],
        "params": int(params),
        "context": int(C),
        "d_model": int(dyn.d_model),
        "n_layers": int(dyn.n_layers),
        "seed": ck.get("seed"),
        "step": ck.get("step"),
        "tokens_seen": ck.get("tokens_seen"),
        "epochs": ck.get("epochs"),
        "train_val_loss": ck.get("val_loss"),
        "heldout_loss": ho["loss"],
        "heldout_last_loss": ho["last_loss"],
        "heldout_cold_loss": ho["cold_loss"],
        "heldout_cold_acc": ho["cold_acc"],
        "heldout_targets": int(len(shared["targets"])),
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


def load_probe(specs: list[str], run_root: str = "runs") -> dict:
    """``<run>=<lr>`` specs -> probe rows read from each run's heldout.json."""
    rows = {}
    for spec in specs:
        name, lr = spec.split("=", 1)
        with open(os.path.join(run_root, name, "heldout.json")) as f:
            h = json.load(f)
        rows[name] = {"lr": float(lr), "heldout_loss": h["loss"], "cold_loss": h["cold_loss"],
                      "cold_acc": h["cold_acc"], "tokens_seen": h["tokens_seen"], "epochs": h["epochs"]}
    return rows


def render_probe(probe: dict) -> list[str]:
    if not probe:
        return []
    finite = {n: r for n, r in probe.items() if np.isfinite(r["heldout_loss"])}
    win = min(finite, key=lambda n: finite[n]["heldout_loss"]) if finite else None
    L = ["", "## 8M learning-rate probe", "",
         "Each run is 0.4 epochs at batch 256 with the cosine planned over that budget; the winner "
         "is the lowest held-out loss at matched tokens, and an endpoint winner extends the probe "
         "by one point, as pre-registered. 26M uses the winner times sqrt(384/512).", "",
         "| run | lr | tokens seen | held-out loss | cold loss | cold acc |",
         "|---|---|---|---|---|---|"]
    for n, r in sorted(probe.items(), key=lambda kv: kv[1]["lr"]):
        mark = " **(winner)**" if n == win else ""
        L.append(f"| {n}{mark} | {r['lr']:.3g} | {(r['tokens_seen'] or 0) / 1e6:.1f}M | "
                 f"{r['heldout_loss']:.4f} | {r['cold_loss']:.4f} | {r['cold_acc']:.3f} |")
    if win:
        lr = probe[win]["lr"]
        L += ["", f"Winner {lr:.3g}; 26M learning rate {float('%.3g' % (lr * (384 / 512) ** 0.5)):.3g}."]
    return L


def render(results: dict, groups: dict, probe: dict | None = None) -> str:
    L = ["# Scaling ladder", "",
         "Every rung is scored on its final checkpoint, on the same held-out frames, reference "
         "trajectory, revisit pairs and hold-still starts, and every context length predicts "
         "the same target frames, so rows are paired comparisons. Read every difference against "
         "the seed spread at the bottom, which the pre-registration defines as the ladder's "
         "resolution. Rules and their branches are in [LADDER_PREREG.md](LADDER_PREREG.md).", "",
         "| run | params | ctx | tokens seen | epochs | held-out loss | cold acc | one-step moving | "
         "copy | ceiling | **headroom** | lead k=1 | k=8 | k=16 | k=32 | return-to-place "
         "(game) | still k=2-10 ident / churn | still k>10 ident |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        ld = r["lead"]
        L.append(
            f"| {r['name']} | {r['params'] / 1e6:.1f}M | {r['context']} | "
            f"{(r['tokens_seen'] or 0) / 1e9:.2f}B | {(r['epochs'] or 0):.2f} | "
            f"{r['heldout_loss']:.3f} | {r['heldout_cold_acc']:.3f} | "
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
              f"Return-to-place is shown as model (game ceiling) over {any_r['revisit_pairs']} revisit "
              f"pairs. Headroom is on the moving subset: (model - copy) / (ceiling - copy). Lead is "
              f"model minus frozen-frame PSNR. Held-out loss is the training objective on "
              f"{any_r['heldout_targets']} fixed validation frames (`python -m ngx.eval.heldout`)."]
    L += ["", "## Seed spread, the ladder's resolution", ""]
    for g, names in groups.items():
        rs = [results[n] for n in names if n in results]
        if len(rs) < 2:
            continue
        v = [r["one_step_moving"] for r in rs]
        h = [r["headroom_pct"] for r in rs]
        lo = [r["heldout_loss"] for r in rs]
        L.append(f"- **{g}** ({len(rs)} runs): one-step moving PSNR "
                 f"{min(v):.2f} to {max(v):.2f} dB, **spread {max(v) - min(v):.2f} dB**; "
                 f"headroom {min(h):.0f}% to {max(h):.0f}%, spread {max(h) - min(h):.0f} points; "
                 f"held-out loss {min(lo):.3f} to {max(lo):.3f}.")
    L += render_probe(probe or {})
    L += ["", "Regenerate with `python -m ngx.eval.ladder --config <rung.yaml> --runs <names> "
          "--group <label>` (and `--probe <run>=<lr> ...` for the probe table); results "
          "accumulate in `docs/ladder_results.json`.", ""]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--runs", nargs="*", default=[], help="run names under runs/")
    p.add_argument("--probe", nargs="*", default=[], help="<run>=<lr> learning-rate probe runs")
    p.add_argument("--group", default=None, help="label for the seed-spread line")
    p.add_argument("--device", default="auto")
    p.add_argument("--windows", type=int, default=800)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--drift-steps", type=int, default=1000)
    p.add_argument("--heldout-targets", type=int, default=4096)
    p.add_argument("--out", default="docs/LADDER.md")
    p.add_argument("--results", default=RESULTS, help="json store; results accumulate here")
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    store = (json.load(open(a.results)) if os.path.exists(a.results)
             else {"results": {}, "groups": {}})
    # Rows scored under an older protocol are not comparable; keep them out.
    for n in [n for n, r in store["results"].items() if r.get("protocol") != PROTOCOL]:
        print(f"{n}: stored under an older eval protocol, dropped until rescored")
        del store["results"][n]
    shared: dict = {}

    for name in a.runs:
        if name in store["results"] and not a.recompute:
            print(f"{name}: cached")
            continue
        print(f"{name}: evaluating ...", flush=True)
        r = evaluate_run(cfg, name, device, shared, a)
        store["results"][name] = r
        print(f"  one-step moving {r['one_step_moving']:.2f} dB (copy {r['copy_moving']:.2f}, "
              f"ceiling {r['ceiling_moving']:.2f}) -> headroom {r['headroom_pct']:.0f}% | "
              f"held-out {r['heldout_loss']:.4f} cold acc {r['heldout_cold_acc']:.3f} | "
              f"tokens {(r['tokens_seen'] or 0) / 1e9:.2f}B epochs {(r['epochs'] or 0):.2f} | lead "
              + " ".join(f"k{k}:{r['lead'][str(k)]:+.2f}" for k in CL_MARKS)
              + f" | r2p {r['revisit_model']:.2f}/{r['revisit_game']:.2f} ({r['revisit_pairs']} pairs) | "
              f"still2-10 {100 * r['still_k2_10_ident']:.0f}%/{r['still_k2_10_churn']:.1f} "
              f"k>10 {100 * r['still_k10_ident']:.1f}%", flush=True)
    if a.group and a.runs:
        store["groups"][a.group] = list(a.runs)
    if a.probe:
        store["probe"] = load_probe(a.probe, cfg.get("run_root", "runs"))

    os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    with open(a.results, "w") as f:
        json.dump(store, f, indent=1)
    with open(a.out, "w") as f:
        f.write(render(store["results"], store["groups"], store.get("probe")))
    print(f"\nwrote {a.out} and {a.results}")


if __name__ == "__main__":
    main()
