# Task packages

The paper entry points are `scripts/task_a.py`, `scripts/task_b.py`, `scripts/task_c.py`, and `scripts/task_e.py`. Run them from the repository root. Their default paper input is the sampling-2 `dataset_v3/` directory; see the [reproduction guide](../docs/reproduction.md).

| Task | Implementation | Sampling-2 input |
| --- | --- | --- |
| A: initial-condition stability | `task_a/main.py` | `ICs_sampling_2.hdf5` and completion table |
| B: prefix stability | `task_b/main.py` | sampling-2 trajectory shards and completion table |
| C: survival | `task_c/survival.py` | sampling-2 trajectory shards and completion table |
| E: forecasting | `task_e/main.py`, `task_e/prepare.py` | sampling-2 trajectory shards and completion table |

`common/` supplies orbital schema, complete-system splitting, evaluation, training helpers, and a read-only input preflight check. A/B/C share the same sampling-2 system split convention. `task_b/memmap.py` supplies a memory-efficient sampling-2 prefix cache for Task B. Shared observation-mask utilities support Task E forecasting experiments.

The source project also contained sampling-3 experiments; the sampling-3-only auxiliary runners were excluded. The sampling-2 dataset is external; no data or trained weights are stored in this repository.
