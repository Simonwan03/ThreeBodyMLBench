# Benchmarking Machine Learning for Chaotic Three-Body Dynamics in Astrophysics

Code for the NeurIPS 2026 Sim2Science paper. Run all commands below from the repository root.

**Dataset — a main contribution of this work:** [Download the public sampling-2 three-body dataset (CITA)](https://www.cita.utoronto.ca/~xuduo/downloads_3body/sampling_2/). The released experiments use its low-resolution data. Data and model checkpoints are not stored in Git.

## 1. Create the environment

The pinned `requirement.txt` records a Linux x86_64, Python 3.12, CUDA 13.0 environment. Create a fresh environment and install its PyTorch wheels from the official CUDA index before the remaining requirements:

```bash
conda create -n threebody-paper python=3.12 -y
conda activate threebody-paper
python -m pip install 'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirement.txt
```

CUDA 13.0 is needed for the pinned GPU wheels; the CPU smoke test below also runs on a compatible existing PyTorch environment. The repository does not contain a lockfile. Record `python -m pip freeze`, GPU type, and driver version with reported results.

The Python packages for the reported paper models are in `requirement.txt`. Model weights and PANDA source code are installed separately in the next section.

## 2. Install foundation model code and weights

Chronos-2 (`chronos-forecasting==2.1.0`), TOTO-2 (`toto-2==2.0.0`), and TiRex-2 (`tirex-2==0.2.1`) are listed in `requirement.txt` and installed by the command above. Their paper checkpoints are `amazon/chronos-2`, `Datadog/Toto-2.0-22m`, and `NX-AI/TiRex-2`. Download them from Hugging Face and verify one 512→128 forecast per model with:

```bash
python scripts/check_foundation_models.py --allow-download
```

PANDA needs its source checkout at the exact commit used for the paper. Clone it, then download `GilpinLab/panda-72M` and check both its Task B encoder and Task E forecaster:

```bash
mkdir -p experiments/foundation_models/vendor
git clone https://github.com/abao1999/panda.git experiments/foundation_models/vendor/panda
git -C experiments/foundation_models/vendor/panda checkout c229e7c8c49cbe294458c44160248bb17856a715
python scripts/check_panda.py --allow-download --device cuda
```

Do **not** run `pip install` inside the PANDA checkout. Its declared NumPy and Transformers versions conflict with `requirement.txt`; this repository imports the pinned source directly. The imports, cached checkpoint, and one encoder/forecast call passed in the existing `stellar` environment. A fresh Python 3.12 installation remains unverified. Use `--device cpu` for the PANDA check on a machine without CUDA.

Once downloaded, weights are reused from the Hugging Face cache. Run the checks again without `--allow-download` to verify they work offline. The full `scripts/task_e.sh` also uses cached weights by default; set `STELLAR_ALLOW_MODEL_DOWNLOAD=1` to let it fetch any missing checkpoint. TiRex-2 uses CPU in the checker by default. Its CUDA path requires an Ampere-or-newer GPU (compute capability 8.0+) and a matching CUDA Toolkit with `nvcc`; on older GPUs, run the full model with `STELLAR_TIREX_DEVICE=cpu bash scripts/task_e.sh tirex2 dataset_v3`.

TimesFM is an auxiliary model outside paper Table 14. Install `requirements-optional.txt` only if you run that model:

```bash
python -m pip install -r requirements-optional.txt
```

See the [foundation-model reproduction details](docs/reproduction.md#foundation-model-setup) for the package and checkpoint map.

## 3. Get the dataset

The linked dataset totals about 103 GB. The downloader fetches the initial conditions, completion table, 20 result shards, and metadata into `dataset_v3/`; it resumes interrupted downloads and checks inputs after completion.

```bash
python scripts/download_sampling_2.py
python scripts/download_sampling_2.py --verify-only
```

To use an existing dataset in another directory, pass `--dataset-dir /path/to/sampling_2` to the downloader and the smoke test. The full Task E shell script accepts that directory as its second argument. A/B/C accept `--dataset-dir`; Task E's Python entry point accepts `--data_glob` and `--completion_table`. Input data can be read-only: generated caches, the common split, and experiment outputs go under `results/`.

## 4. Check the end-to-end pipeline

```bash
bash scripts/smoke_test.sh
# or: bash scripts/smoke_test.sh /path/to/sampling_2
```

This uses small real-data subsets and CPU baselines for A, B, C, and E. It writes metrics under `results/smoke/`. These metrics **are not the paper results**.

## 5. Run the sampling-2 experiments

The commands below use all eligible data and the paper-mapped settings. They can take substantial time and memory. Task E prepares a reusable window cache on its first run.

```bash
python scripts/task_a.py --resolution low --target-mode stable_fraction \
  --models equation2,equation4,mlp,xgboost --epochs 50 \
  --output-dir results/task_a/sampling_2/fraction
python scripts/task_a.py --resolution low --target-mode binary_any \
  --models equation2,equation4,mlp,xgboost --epochs 50 \
  --output-dir results/task_a/sampling_2/binary
python scripts/task_b.py --resolution low --prefix-len 96 \
  --models svm,cnn,tcn --output-dir results/task_b/sampling_2/prefix96
python scripts/task_b.py --resolution low --prefix-len 512 \
  --models svm,cnn,tcn --output-dir results/task_b/sampling_2/prefix512
python scripts/task_c.py --resolution low --prefix-len 96 --hazard-t-min 5 \
  --models baseline,stats,cnn,tcn --epochs 20 --batch-size 128 \
  --output-dir results/task_c/sampling_2/prefix96_raw
bash scripts/task_e.sh itransformer dataset_v3
```

For the Task E foundation baselines, run `bash scripts/task_e.sh chronos2 dataset_v3`, `bash scripts/task_e.sh panda_zero_shot dataset_v3`, `bash scripts/task_e.sh toto2 dataset_v3`, or `bash scripts/task_e.sh tirex2 dataset_v3` after completing [foundation model setup](#2-install-foundation-model-code-and-weights).

For the Task B foundation probes and model-specific options, see the [reproduction guide](docs/reproduction.md), [task map](tasks/README.md), and each command's `--help`. Results and run settings are saved in each output directory (`config.json`, `metrics.json`, plus predictions and model artifacts where applicable).

The [reproduction guide](docs/reproduction.md) maps paper tables 8–14 to entry points and records external inputs. Full training and foundation-model numerical agreement remain unchecked. The full-data analytic baselines in Tables 8 and 9 match the paper to its reported four decimals.

## Code map

| Path | Role |
| --- | --- |
| `scripts/task_a.py`, `task_b.py`, `task_c.py`, `task_e.py` | Paper task entry points |
| `scripts/task_e.sh` | Task E preparation and model run sequence |
| `scripts/download_sampling_2.py` | Resumable dataset download and input check |
| `scripts/smoke_test.sh` | Small end-to-end checks on real data |
| `tasks/` | Data loading, models, metrics, and shared utilities |
| `experiments/foundation_models/` | Task B Panda, Chronos-2, and XGBoost probes |

## Computational Resources

The experiments were primarily conducted on an **Amazon AWS EC2 `g7.2xlarge` instance**, equipped with:

- **GPU:** NVIDIA RTX PRO 4500 Blackwell (32 GB VRAM)
- **CPU:** 8 vCPUs (Intel Xeon)
- **Memory:** 32 GiB RAM
- **Storage:** 440 GB local NVMe SSD

This instance was primarily used for training and evaluating supervised learning models and pretrained time-series foundation models. Computational requirements vary depending on the prediction task, model architecture, and experimental configuration.

## License

MIT; see [LICENSE](LICENSE).
