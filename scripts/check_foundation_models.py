"""Check the non-PANDA paper foundation models with one 512->128 forecast each."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tasks.task_e.foundation import (  # noqa: E402
    build_chronos2_predictor,
    build_tirex2_predictor,
    build_toto2_predictor,
)

BUILDERS = {
    "chronos2": (build_chronos2_predictor, "amazon/chronos-2"),
    "toto2": (build_toto2_predictor, "Datadog/Toto-2.0-22m"),
    "tirex2": (build_tirex2_predictor, "NX-AI/TiRex-2"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*BUILDERS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu.")

    time = np.arange(512, dtype=np.float32)[None, :, None]
    channel = np.arange(15, dtype=np.float32)[None, None, :]
    context = np.sin(time / 30.0 + channel / 7.0).astype(np.float32)
    selected = tuple(BUILDERS) if args.model == "all" else (args.model,)
    for name in selected:
        builder, model_id = BUILDERS[name]
        if args.device == "auto":
            # TiRex-2 needs a matching CUDA Toolkit and nvcc on its CUDA path.
            device = "cpu" if name == "tirex2" else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = args.device
        if name == "tirex2" and device == "cuda":
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                parser.error(
                    f"TiRex-2 CUDA requires compute capability >= 8.0, got {capability}; "
                    "use --device cpu."
                )
        options = SimpleNamespace(
            chronos_model_id=model_id if name == "chronos2" else "amazon/chronos-2",
            toto2_model_id=model_id if name == "toto2" else "Datadog/Toto-2.0-22m",
            tirex2_model_id=model_id if name == "tirex2" else "NX-AI/TiRex-2",
            hf_local_files_only=not args.allow_download,
            torch_dtype="float32",
            context_len=512,
            batch_size=1,
        )
        print(f"CHECK {name}: checkpoint={model_id}, device={device}", flush=True)
        predictor = builder(options, device)
        prediction = predictor(context, 128)
        if prediction.shape != (1, 128, 15) or not np.isfinite(prediction).all():
            raise RuntimeError(f"{name} returned invalid forecast: shape={prediction.shape}")
        print(f"PASS {name}: 512->128 forecast with 15 channels", flush=True)
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
