"""Check the sampling-2 dataset_v3 inputs used by the paper tasks.

The check is read-only. It verifies file presence and the sampling-2 shard
layout; the task entry points create or validate their shared system split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(task: str, dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    if not dataset_dir.is_dir():
        return [f"Data directory is missing or its symlink is broken: {dataset_dir}"]

    completion = dataset_dir / "completed_systems_sampling_2.txt"
    if not completion.is_file():
        errors.append(f"Missing completion table: {completion}")

    if task == "A":
        ics = dataset_dir / "ICs_sampling_2.hdf5"
        if not ics.is_file():
            errors.append(f"Missing initial conditions: {ics}")
        return errors

    shards = sorted(dataset_dir.glob("results_batch_*_sampling_2.hdf5"))
    expected_names = {f"results_batch_{number:02d}_sampling_2.hdf5" for number in range(20)}
    actual_names = {path.name for path in shards}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        errors.append(f"Expected 20 sampling-2 shards; missing={missing}, extra={extra}")

    if shards:
        try:
            import h5py
        except ImportError:
            errors.append("h5py is unavailable; install the project environment")
        else:
            for path in shards:
                try:
                    with h5py.File(path, "r") as handle:
                        shape = tuple(handle["times_complete"].shape)
                    if shape != (8000, 2001):
                        errors.append(f"Unexpected times_complete shape {shape}: {path}")
                except (OSError, KeyError) as exc:
                    errors.append(f"Cannot read times_complete in {path}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list("ABCE"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_v3"))
    args = parser.parse_args()
    errors = check(args.task, args.dataset_dir)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: Task {args.task} sampling-2 inputs checked under {args.dataset_dir}")
    print("System split, pretrained weights, and full training/evaluation remain unchecked.")


if __name__ == "__main__":
    main()
