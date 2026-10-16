#!/usr/bin/env bash
# Quick end-to-end checks on a small subset of the real sampling-2 dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
data_dir=${1:-dataset_v3}
python scripts/download_sampling_2.py --verify-only --dataset-dir "$data_dir"
python scripts/task_a.py --resolution low --dataset-dir "$data_dir" \
  --models equation2 --max-cases 80 --device cpu --output-dir results/smoke/task_a
python scripts/task_b.py --resolution low --dataset-dir "$data_dir" \
  --max-shards 1 --rows-per-shard 640 --prefix-len 100 \
  --models svm --epochs 1 --batch-size 64 --device cpu --output-dir results/smoke/task_b
python scripts/task_c.py --resolution low --dataset-dir "$data_dir" \
  --max-shards 1 --rows-per-shard 640 --prefix-len 100 \
  --models baseline --device cpu --output-dir results/smoke/task_c
python scripts/task_e.py --resolution low \
  --data_glob "$data_dir/results_batch_00_sampling_2.hdf5" \
  --completion_table "$data_dir/completed_systems_sampling_2.txt" \
  --model linear_extrapolation --representation orbital15 \
  --context_len 512 --pred_len 128 --max_trajectories 320 \
  --max_windows_per_trajectory 1 --train-windows-per-trajectory 1 \
  --val-windows-per-trajectory 1 --test-windows-per-trajectory 1 \
  --bootstrap-samples 0 --num_workers 0 --device cpu \
  --output_dir results/smoke/task_e
printf 'PASS: smoke outputs are in results/smoke/\n'
