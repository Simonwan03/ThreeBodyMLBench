# Benchmarking Machine Learning for Chaotic Three-Body Dynamics in Astrophysics

Code for the NeurIPS 2026 Sim2Science paper, selected from `stellar_dynamics_project`. The paper experiments in this release use the low-resolution sampling-2 dataset at `dataset_v3/`.

## Code map

| Area | Purpose |
| --- | --- |
| `scripts/task_a.py`, `task_b.py`, `task_c.py`, `task_e.py` | Paper task entry points; their defaults use sampling 2 |
| `scripts/task_e.sh` | Task E preparation and model run sequence using `dataset_v3/` |
| `scripts/download_sampling_2.py` | Download and validate sampling-2 inputs at `dataset_v3/` |
| `tasks/` | Data loading, models, evaluation, and shared utilities |
| `experiments/foundation_models/` | Task B Panda and Chronos-2 probes and dynamical-summary XGBoost |

See the [task map](tasks/README.md) and [reproduction guide](docs/reproduction.md). All included experiment commands default to `dataset_v3` and sampling ID 2. Sampling-3-only auxiliary experiments were excluded from this release.

## Data and setup

The paper input is available from the [CITA sampling-2 download directory](https://www.cita.utoronto.ca/~xuduo/downloads_3body/sampling_2/). From the repository root, install `requirement.txt` in an environment suited to your hardware, then run:

```bash
python -m pip install -r requirement.txt
python scripts/download_sampling_2.py
```

The downloader writes to `dataset_v3/` (the default input path for all included paper scripts). It fetches `ICs_sampling_2.hdf5`, `completed_systems_sampling_2.txt`, `results_batch_00_sampling_2.hdf5` through `results_batch_19_sampling_2.hdf5`, and `metadata.txt`. It skips files whose sizes match the server, resumes partial downloads in `.part` files, and checks the downloaded inputs for Tasks A, B, C, and E. The full dataset is about 103 GB; keep enough free space. Rerun the command after an interruption. On this machine, `dataset_v3` is already a local link to `/data/sampling_2`; on a fresh checkout the command creates the directory. Data files are ignored by Git.

To check an existing copy without network access, run `python scripts/download_sampling_2.py --verify-only`. A/B/C generate or validate a shared system split from the sampling-2 completion table. No sampling-3 split is bundled here.

Pretrained Panda, Chronos-2, and Toto-2 weights are external. Panda runs in Tasks B and E also require the upstream Panda source at `experiments/foundation_models/vendor/panda`; the source project used commit `c229e7c8c49cbe294458c44160248bb17856a715` from `https://github.com/abao1999/panda.git`.

Check the inputs and inspect a task command:

```bash
python -m tasks.common.preflight --task B --dataset-dir dataset_v3
python scripts/task_b.py --resolution low --help
```

For example, run the sampling-2 prefix-stability baseline:

```bash
python scripts/task_b.py --resolution low \
  --dataset-dir dataset_v3 \
  --prefix-len 100 --models svm,cnn,tcn \
  --output-dir results/task_b/sampling_2/reproduce_task_b
```

The raw data, pretrained weights, embedding caches, checkpoints, and generated results are not included. Exact numerical reproduction also requires the settings for each reported run.

## Code availability

The repository URL for the paper is [github.com/Simonwan03/Benchmarking-Machine-Learning-for-Chaotic-Three-Body-Dynamics-in-Astrophysics](https://github.com/Simonwan03/Benchmarking-Machine-Learning-for-Chaotic-Three-Body-Dynamics-in-Astrophysics).

## License

MIT; see [LICENSE](LICENSE).
