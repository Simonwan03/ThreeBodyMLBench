"""Task C sampling-2 survival-curve entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_c.survival import main as task_main


def main() -> None:
    run_entry(
        task="C", selected_main=task_main, default_resolution="low",
        supported=("low",),
        defaults={"low": (
            "--dataset-dir", "dataset_v3",
            "--output-dir", "results/task_c/sampling_2/task_c_survival20",
        )},
    )


if __name__ == "__main__":
    main()
