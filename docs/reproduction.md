# Sampling-2 paper code map

Run commands from the repository root with the external `dataset_v3/` sampling-2 dataset. Every documented input path below points to that dataset. Sampling-3-only auxiliary experiments from the development project are excluded from this repository.

| Task | Entry point | Target |
| --- | --- | --- |
| A | `python scripts/task_a.py --resolution low --dataset-dir dataset_v3 --help` | System stable fraction from initial conditions |
| B | `python scripts/task_b.py --resolution low --dataset-dir dataset_v3 --help` | Trajectory stability from early prefixes |
| C | `python scripts/task_c.py --resolution low --dataset-dir dataset_v3 --help` | Multi-horizon survival prediction |
| E | `bash scripts/task_e.sh <model> dataset_v3` | Trajectory forecasting; `<model>` is `chronos2`, `panda_zero_shot`, `toto2`, `itransformer`, or `tcn` |

Task E's single-run Python entry point is `python scripts/task_e.py --resolution low --output_dir results/task_e/sampling_2/<run_name>`; it uses `dataset_v3/results_batch_*_sampling_2.hdf5` and `dataset_v3/completed_systems_sampling_2.txt` by default.

## Data and split

Download the initial-condition file, completion table, and 20 result shards from the [CITA sampling-2 directory](https://www.cita.utoronto.ca/~xuduo/downloads_3body/sampling_2/) with `python scripts/download_sampling_2.py`. The files land in `dataset_v3/`, which all paper entry points use by default. The downloader resumes interrupted `.part` files and runs input preflight after completion. For an existing copy, use `python scripts/download_sampling_2.py --verify-only`; this performs the input checks offline. To store data elsewhere, run `python scripts/download_sampling_2.py --dataset-dir /path/to/data` and link that directory at the repository root with `ln -s /path/to/data dataset_v3` before running the paper commands.

Tasks A, B, and C use the same complete-system split convention: 32 variations per `raw_case_id`, 70/15/15 train/validation/test, seed 42, and horizon 10,000. Their code creates or validates `dataset_v3/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv` from the completion table. The manifest is generated from the completion table when a task first needs it; it is not bundled in Git. Task E uses its own system-level split implementation.

## Task B foundation baselines

The Panda probe, Chronos-2 cache and linear probe, and dynamical-summary XGBoost scripts under `experiments/foundation_models/` default to `dataset_v3` and sampling ID 2. Their generated labels and embeddings must match the same Task B split and row order. Panda source and model weights, Chronos-2 weights, and caches are separate inputs.

## Verification limit

This machine has the sampling-2 data at `/data/sampling_2`, linked locally as `dataset_v3`. Input preflight passes for Tasks A, B, C, and E. Full training and numerical agreement with the paper have not been checked.
