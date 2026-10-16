# Sampling-2 cross-task comparisons

These auxiliary analyses use saved predictions from the `dataset_v3` sampling-2 experiments:

| Module | Purpose |
| --- | --- |
| `evaluate_common_target.py` | Compare Task A/B/C predictions on the same system stable-fraction target at horizon 10,000 |
| `future_only_ablation.py` | Evaluate trajectories still at risk after an observed prefix |

Run from the repository root with the matching sampling-2 prediction files. The sampling-3 fixed-budget and factorial experiments from the development project are not part of this paper code release.
