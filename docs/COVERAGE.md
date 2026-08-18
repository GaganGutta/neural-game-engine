# B2: data coverage

Grid cell = 64 map units. **Effective cells** is `exp(entropy)` of the visit distribution: the number of equally-visited cells that would produce the same spread. It is the honest version of 'how much of the map did we see', because distinct-cell counts reward a policy that touches a thousand cells once and then spins in a corner.

| metric | before |
|---|---|
| frames | 150,000 |
| episodes | 352 |
| distinct cells | 87 |
| **effective cells** | **63.4** |
| top-10% of cells hold | 29.0% of visits |
| **action entropy** (1.0 = uniform) | **0.887** |
| rarest action share | 5.4% |

![coverage](..\assets\coverage_before.png)

Top row is visitation on a log colour scale, so a uniform-looking map is genuinely uniform and a few bright cells against a dark field is a policy that parked. Bottom row is the action distribution against the uniform line.

Regenerate with `python -m ngx.eval.coverage --roots data/my_way_home --labels before`.
