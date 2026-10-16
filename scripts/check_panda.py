"""Check the pinned PANDA source and run one checkpoint-backed forecast."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tasks.task_e.panda_dependency import load_panda_pretrained, panda_vendor_root  # noqa: E402

EXPECTED_COMMIT = "c229e7c8c49cbe294458c44160248bb17856a715"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-id", default="GilpinLab/panda-72M")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--source-only", action="store_true", help="Check source without loading weights.")
    args = parser.parse_args()

    vendor = panda_vendor_root()
    if not vendor.is_dir():
        parser.error(f"PANDA source is missing at {vendor}; follow the README clone command.")
    git = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    commit = git.stdout.strip()
    if commit != EXPECTED_COMMIT:
        parser.error(f"PANDA source commit is {commit}; expected {EXPECTED_COMMIT}.")
    from tasks.task_e.panda_dependency import ensure_panda_importable

    ensure_panda_importable()
    from panda.patchtst.patchtst import PatchTSTForPrediction  # noqa: F401

    print(f"PASS: PANDA source import at {commit}")
    if args.source_only:
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu.")
    model = load_panda_pretrained(
        args.model_id, device, local_files_only=not args.allow_download,
    )
    context = torch.zeros((1, 512, 15), dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model.generate(
            past_values=context,
            past_observed_mask=torch.ones_like(context, dtype=torch.bool),
        ).sequences
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[-2:] != (128, 15):
        raise RuntimeError(f"Unexpected PANDA output shape: {tuple(prediction.shape)}")
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("PANDA output contains non-finite values")
    print(f"PASS: PANDA Task E forecast on {device}; shape {tuple(prediction.shape)}")

    from experiments.foundation_models.panda.train_panda_unbound_classifier import pool_encoder_features

    prefix = context[:, :96, :]
    with torch.no_grad():
        hidden = model.model(
            past_values=prefix,
            past_observed_mask=torch.ones_like(prefix),
            return_dict=True,
        ).last_hidden_state
        features = pool_encoder_features(hidden, "patch_mean_keep_channels")
    if features.shape != (1, 15 * 768) or not bool(torch.isfinite(features).all()):
        raise RuntimeError(f"Unexpected PANDA Task B features: {tuple(features.shape)}")
    print(f"PASS: PANDA Task B prefix-96 encoder; shape {tuple(features.shape)}")


if __name__ == "__main__":
    main()
