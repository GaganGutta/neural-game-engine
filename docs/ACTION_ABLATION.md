# B0.5b: action conditioning

6 independent contexts, 6 distinct actions each held for 16 frames, greedy decoding so the action is the only thing that varies.

## Divergence: do different buttons produce different futures?

Mean pairwise PSNR *between* branches. **Lower means more separation.**

| condition | mean pairwise PSNR |
|---|---|
| real game | **17.36 dB** |
| model | **18.40 dB** |
| model, action embedding zeroed | 60.00 dB (clipped at 60) |

The model separates futures about as much as the real game does. The zeroed row is the load-bearing control: identical input makes the branches pixel-identical, so it pins at the clipping cap and proves the harness really is varying the action.

No percentage is quoted here. Expressing the model as a fraction of the distance from the zeroed control to the real game would put the clipping constant in the denominator, and the headline would move whenever that constant moved.

## Grounding: are they the *right* different futures?

Divergence is necessary and not sufficient: a model branching at random scores the same. This compares the model's rollout under action `a` to the real game's rollout under `a`, and to the real game's rollouts under every other action.

| comparison | mean PSNR |
|---|---|
| model under `a` vs **real under `a`** (matched) | **20.45 dB** |
| model under `a` vs real under `b != a` (mismatched) | 16.79 dB |
| margin | **+3.66 dB** (6/6 contexts favour matched) |

**Grounded.**

![opposing actions from one context](action_ablation.gif)

Both panes start from the same context and then hold opposing turn actions. The PSNR readout is between the panes, not against ground truth.

Regenerate with `python -m ngx.eval.action_ablation --config configs/small.yaml`.
