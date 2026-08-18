# B0.5b: action-conditioning ablation

6 independent contexts, 6 distinct actions each held for 16 frames, greedy decoding so the action is the only thing that varies. The number reported is mean pairwise PSNR *between* branches: **lower means the actions produced more different futures.**

| condition | mean pairwise PSNR | meaning |
|---|---|---|
| real game | **17.36 dB** | how much the actions truly separate the future |
| model | **18.40 dB** | how much the model separates them |
| model, action embedding zeroed | 60.00 dB | control: identical input, so branches are pixel-identical and PSNR pins at the 60 dB cap |

**The model listens to the controller.**

Placing the model on the scale between the two references: it recovers **98%** of the real game's action separation. 0% would mean the button does nothing; 100% would mean it separates futures exactly as much as VizDoom does.

The zeroed row is the load-bearing control. It pins at the cap, which proves the harness really is varying the action and that the measured model number is not an artifact of the branches sharing a context.

![opposing actions from one context](action_ablation.gif)

Both panes start from the same 6 frames of real context and then hold opposing turn actions. The PSNR readout is between the panes, not against ground truth.

Regenerate with `python -m ngx.eval.action_ablation --config configs/small.yaml`.
