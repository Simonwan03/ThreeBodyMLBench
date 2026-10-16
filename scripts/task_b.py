"""Task B sampling-2 prefix-stability entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_b.main import main as task_main


def main() -> None:
    run_entry(
        task="B", selected_main=task_main, default_resolution="low",
        supported=("low",), defaults={"low": ("--dataset-dir", "dataset_v3")},
    )


if __name__ == "__main__":
    main()
