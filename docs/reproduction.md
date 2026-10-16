# Sampling-2 paper reproduction guide

Start with the environment, [public data](https://www.cita.utoronto.ca/~xuduo/downloads_3body/sampling_2/), and smoke test in the [README](../README.md). Run every command from the repository root. The paper PDF used for this map is the 22-page draft in the companion `stellar_dynamics_project` workspace; the table numbers below refer to that draft.

## Shared input and split

```bash
python scripts/download_sampling_2.py --verify-only
bash scripts/smoke_test.sh
```

Tasks A/B/C use one system-level split: 4,998 complete systems, 32 variations each, 70/15/15 train/validation/test, seed 42, and stability horizon 10,000. The code generates and validates `results/splits/common_case_split_sampling_2_g32_h10000_r70-15-15_seed42.csv` (3,498/750/750 systems). Pass `--split-manifest` to all three tasks if a different writable location is required. Task A's tokenized HDF5 is generated in `results/cache/task_a/`; pass `--tokenized-path` to relocate it. The original input directory can be read-only.

Task E filters trajectories that can form a complete 512+128 window, then uses a separate system-level split. Its records and training-only scaler live in `results/task_e/sampling_2/reproduce_task_e_prepared/` when run through the shell script. The cache loader checks data files and preparation parameters against its manifest. If changing those parameters, use a different cache directory or remove the old one.

## Paper table map

| Draft table | Method and input | Command or entry point | Expected result |
| --- | --- | --- | --- |
| 8, binary all-stable | Equation (2), Equation (4), MLP, XGBoost | Task A `--target-mode binary_any --models equation2,equation4,mlp,xgboost --epochs 50` | `metrics.json`: Equation (2) accuracy 0.6187; Equation (4) accuracy 0.6600; XGBoost accuracy 0.8053 |
| 9, stable fraction | Same local methods | Task A `--target-mode stable_fraction --models equation2,equation4,mlp,xgboost --epochs 50` | Equation (2) MSE 0.2488; Equation (4) MSE 0.2268; XGBoost MSE 0.0884 |
| 8–9, pretrained MLP | Official *Ghost* MLP checkpoint | Task A `--models pretrained_ghost_mlp`, separately for each target mode | Binary AUROC 0.9086; fraction MAE 0.2176 |
| 10, raw prefix | `orbital15`, prefix 96 and 512 | Task B `--prefix-len 96` or `512` with `--models svm,cnn,tcn` | See the paper's six raw-sequence rows |
| 10, summary features | 120-feature dynamical summary, XGBoost | `experiments/foundation_models/shared/train_spock_inspired_task_b_xgboost.py` using the matching Task B prefix directory | AUROC 0.9649 at 96; 0.9851 at 512 |
| 10, frozen encoders | PANDA and Chronos-2 linear probes | Scripts in `experiments/foundation_models/panda/` and `chronos2/` using matching Task B labels and embedding caches | PANDA/Chronos-2 AUROC 0.9467/0.9531 at 96, 0.9786/0.9791 at 512 |
| 11–12, at-risk cohorts | Existing A/B predictions, filtered by prefix endpoint | `tasks/comparisons/evaluate_common_target.py` and `future_only_ablation.py` | See paper for cohort sizes and metrics |
| 13, raw survival | Prefix 96; 20 log bins from 5 to 10,000 | Task C `--models baseline,stats,cnn,tcn --prefix-len 96 --hazard-t-min 5 --epochs 20 --batch-size 128` | Baseline overall MAE 0.3099; raw TCN 0.0662 |
| 13, frozen survival | PANDA and Chronos-2 aligned embedding caches | Task C `--embedding-cache <aligned.npy> --embedding-labels <aligned.csv> --models baseline,embedding_direct,embedding_meanstd,embedding_attention` | Chronos-2 direct overall MAE 0.0376, IBS 0.0416 |
| 14, one-shot forecasting | Chronos-2, PANDA, iTransformer, TOTO2, TiRex2 | `bash scripts/task_e.sh <model> dataset_v3` | Chronos-2 MAE(z) 0.2282; iTransformer MSE(z) 0.2123 |

### A: full-data commands

```bash
python scripts/task_a.py --resolution low --target-mode binary_any \
  --models equation2,equation4,mlp,xgboost --epochs 50 \
  --output-dir results/task_a/sampling_2/binary
python scripts/task_a.py --resolution low --target-mode stable_fraction \
  --models equation2,equation4,mlp,xgboost --epochs 50 \
  --output-dir results/task_a/sampling_2/fraction
```

The paper's Vynatheya MLP row uses the **Ghost** variant. Obtain the upstream [`mlp_model_trip_ghost_v1.2.2.pkl`](https://raw.githubusercontent.com/pavanvyn/triple-stability/main/mlp_model_trip_ghost_v1.2.2.pkl) and pass `--pretrained-ghost-mlp-path /path/to/mlp_model_trip_ghost_v1.2.2.pkl --models pretrained_ghost_mlp` for both Task A target modes. The loader verifies its SHA-256 (`a260845fc9c5b80ede3ab0028ac582bffa8450dad0c23f3ce78d73987932b866`). The standard non-Ghost MLP has different paper-external results and is not the row in Tables 8–9.

### B: raw and summary runs

Run the full Task B prefix with each reported length, then use its `labels_filtered.csv` and `split_indices.npz` for the associated feature and foundation runs:

```bash
python scripts/task_b.py --resolution low --prefix-len 96 --models svm,cnn,tcn \
  --output-dir results/task_b/sampling_2/prefix96
python scripts/task_b.py --resolution low --prefix-len 512 --models svm,cnn,tcn \
  --output-dir results/task_b/sampling_2/prefix512
python experiments/foundation_models/shared/train_spock_inspired_task_b_xgboost.py \
  --dataset-dir dataset_v3 --sampling-id 2 --prefix-steps 96 \
  --prefix-dir results/task_b/sampling_2/prefix96 \
  --output-dir results/task_b/sampling_2/summary96
```

Repeat the summary command with `512` and `prefix512`. For frozen encoders, use `train_panda_task_b_split.py` with `--prefix-dir`, `--prefix-steps`, `--checkpoint GilpinLab/panda-72M`, and `--head linear`; use `run_chronos2_task_b_cache.py` then `train_chronos2_task_b_linear.py` with matching labels, prefix length, and cache. Both need external model weights and are not covered by the smoke test. Check the output cache metadata and label fingerprints before training; a cache from the other prefix length must not be reused.

### C: survival runs

```bash
python scripts/task_c.py --resolution low --prefix-len 96 --hazard-t-min 5 \
  --hazard-t-max 10000 --hazard-bins 20 --variation-target full32 \
  --models baseline,stats,cnn,tcn --epochs 20 --batch-size 128 \
  --output-dir results/task_c/sampling_2/prefix96_raw
```

The foundation rows additionally need aligned PANDA or Chronos-2 embedding caches. Supply their `.npy` file via `--embedding-cache` and the matching labels via `--embedding-labels`; run the three `embedding_*` heads with `--epochs 20 --prefix-len 96 --hazard-t-min 5`. The Task C code validates cache length and fingerprints where metadata is available. `tasks.task_c.main` is an older seven-horizon objective; the paper entry point `scripts/task_c.py` calls the 20-bin event-time survival implementation.

### E: forecasting runs

```bash
bash scripts/task_e.sh chronos2 dataset_v3
bash scripts/task_e.sh panda_zero_shot dataset_v3
bash scripts/task_e.sh itransformer dataset_v3
bash scripts/task_e.sh toto2 dataset_v3
bash scripts/task_e.sh tirex2 dataset_v3
```

The shell script prepares all eligible trajectories, then uses 8 sampled training, 4 validation, 4 E1 test, and 2 rolling-start windows per trajectory; context/prediction lengths are 512/128, stride is 20, and E2 rolls four blocks. These settings match the saved paper protocol. iTransformer uses a 30-epoch run, learning rate 0.0003, 2 layers, dimension 256, and 8 heads. `tcn` is also accepted by the shell script as an auxiliary model but is not a Table 14 row.

Foundation checkpoints are external. Set `STELLAR_ALLOW_MODEL_DOWNLOAD=1` to permit downloads or populate the Hugging Face cache first. PANDA additionally imports the [upstream source](https://github.com/abao1999/panda) at commit `c229e7c8c49cbe294458c44160248bb17856a715`; clone it to `experiments/foundation_models/vendor/panda` or set `STELLAR_PANDA_VENDOR_ROOT`.

## Foundation model setup

| Paper model | Python package in `requirement.txt` | Hugging Face checkpoint | Local verification |
| --- | --- | --- | --- |
| Chronos-2 | `chronos-forecasting==2.1.0` | `amazon/chronos-2` | `python scripts/check_foundation_models.py --model chronos2` |
| PANDA | Source checkout at the commit in the README | `GilpinLab/panda-72M` | `python scripts/check_panda.py` |
| TOTO-2 | `toto-2==2.0.0` | `Datadog/Toto-2.0-22m` | `python scripts/check_foundation_models.py --model toto2` |
| TiRex-2 | `tirex-2==0.2.1` | `NX-AI/TiRex-2` | `python scripts/check_foundation_models.py --model tirex2` |

Run `python -m pip install -r requirement.txt` once in the paper environment; no extra pip install is needed for Chronos-2, TOTO-2, or TiRex-2. The checkers load locally cached checkpoints by default and each run one 512→128 forecast. Add `--allow-download` to fetch missing weights. `scripts/task_e.sh` also defaults to cached weights; set `STELLAR_ALLOW_MODEL_DOWNLOAD=1` when running it the first time if the checkpoints are absent. For TiRex-2, the checker defaults to CPU. CUDA inference additionally requires an Ampere-or-newer GPU (compute capability 8.0+), a compatible CUDA Toolkit, and `nvcc` because TiRex-2 compiles a kernel on first use. On an older GPU, set `STELLAR_TIREX_DEVICE=cpu` when invoking `scripts/task_e.sh tirex2`. TimesFM is an auxiliary baseline outside Table 14; install `requirements-optional.txt` only for that path.

## Verification status

The public data URL answered an HTTP HEAD request; all 22 required data files passed the offline input check. A/B/C/E completed small real-data runs through `bash scripts/smoke_test.sh`. On the full 4,998-system split, Task A's analytic baselines reproduced the four-decimal entries in paper Tables 8–9: Equation (2) binary accuracy 0.6187 and fraction MSE 0.2488; Equation (4) binary accuracy 0.6600 and fraction MSE 0.2268. Full learned-model training, embedding-cache generation, full-data Task E foundation evaluation, and their numerical agreement with the draft remain to be checked. Single-window Chronos-2 and TOTO-2 GPU forecasts and a TiRex-2 CPU forecast passed using cached checkpoints. A clean install of the pinned Python 3.12/CUDA 13.0 environment on another machine remains unverified. The pinned PANDA upstream package declares NumPy and Transformers versions that conflict with `requirement.txt`, so this repository imports the checkout directly rather than installing it. Source import, cached checkpoint loading, one 512→128 GPU forecast, and one prefix-96 encoding all passed in the existing `stellar` environment using `python scripts/check_panda.py`; a fresh install remains unverified.
