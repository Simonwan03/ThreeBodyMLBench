"""Task E sampling-2 trajectory-forecasting entry point."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.common.resolution_entry import run_entry
from tasks.task_e.main import main as task_main


def main() -> None:
    run_entry(
        task="E", selected_main=task_main, default_resolution="low",
        supported=("low",), forward_sampling_id=False,
        defaults={"low": (
            "--data_glob", "dataset_v3/results_batch_*_sampling_2.hdf5",
            "--completion_table", "dataset_v3/completed_systems_sampling_2.txt",
        )},
    )


if __name__ == "__main__":
    main()
