# B0.5a: overfit one batch

4 fixed windows from the tokenized dataset, 2.0M-parameter model built from `configs/small.yaml`, lr 0.001, same random cosine masking as real training. The batch never changes.

| step | train loss | masked acc | cold acc |
|---|---|---|---|
| 0 | 6.2872 | 0.001 | 0.023 |
| 50 | 2.0290 | 0.485 | 0.355 |
| 100 | 0.5661 | 0.868 | 0.758 |
| 150 | 0.0694 | 0.998 | 1.000 |
| 200 | 0.0112 | 1.000 | 1.000 |
| 250 | 0.0004 | 1.000 | 1.000 |
| 300 | 0.0001 | 1.000 | 1.000 |
| 350 | 0.0000 | 1.000 | 1.000 |
| 400 | 0.0000 | 1.000 | 1.000 |
| 450 | 0.0000 | 1.000 | 1.000 |
| 500 | 0.0000 | 1.000 | 1.000 |
| 550 | 0.0000 | 1.000 | 1.000 |
| 600 | 0.0000 | 1.000 | 1.000 |
| 650 | 0.0000 | 1.000 | 1.000 |
| 700 | 0.0000 | 1.000 | 1.000 |
| 750 | 0.0534 | 0.984 | 0.984 |
| 800 | 0.0974 | 0.981 | 0.988 |
| 850 | 0.0001 | 1.000 | 1.000 |
| 900 | 0.0000 | 1.000 | 1.000 |
| 950 | 0.0000 | 1.000 | 1.000 |
| 1000 | 0.0000 | 1.000 | 1.000 |

Masked accuracy first cleared 0.99 at step 141. **PASS.** The architecture can memorise a batch, so the loss, the masking, the attention mask and the data pipeline are wired correctly. A weak model on the full dataset is an undertrained or undersized model, not a broken one.

`masked acc` is accuracy on the randomly masked positions, i.e. the training objective. `cold acc` hides the entire target frame, which is what the first MaskGIT pass faces at play time; it is the harder number and it is reported so that memorising via visible neighbours inside the target frame cannot pass as learning the dynamics.

Regenerate with `python -m ngx.train.overfit --config configs/small.yaml`.
