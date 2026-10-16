"""Download the public sampling-2 dataset into the paper scripts' default input path."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.common.preflight import check  # noqa: E402

BASE_URL = "https://www.cita.utoronto.ca/~xuduo/downloads_3body/sampling_2/"
REQUIRED_FILES = (
    "ICs_sampling_2.hdf5",
    "completed_systems_sampling_2.txt",
    *(f"results_batch_{batch:02d}_sampling_2.hdf5" for batch in range(20)),
)
OPTIONAL_FILES = ("metadata.txt",)
CHUNK_SIZE = 8 * 1024 * 1024
MAX_ATTEMPTS = 5


def remote_size(url: str) -> int:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        header = response.headers.get("Content-Length")
    if header is None or int(header) <= 0:
        raise ValueError(f"Server did not provide a valid Content-Length: {url}")
    return int(header)


def download_file(url: str, target: Path, expected_size: int) -> None:
    if target.is_file() and target.stat().st_size == expected_size:
        print(f"SKIP {target.name}: already {expected_size:,} bytes", flush=True)
        return
    if target.exists() and not target.is_file():
        raise ValueError(f"Expected a regular file: {target}")

    partial = target.with_name(target.name + ".part")
    if partial.exists() and not partial.is_file():
        raise ValueError(f"Expected a regular partial file: {partial}")
    if partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            partial.replace(target)
            print(f"DONE {target.name}: {expected_size:,} bytes", flush=True)
            return
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range for {target.name}: {content_range}")
                    mode = "ab"
                elif response.status == 200:
                    mode = "wb"  # The server ignored Range, so restart this file.
                else:
                    raise ValueError(f"Unexpected HTTP {response.status} for {target.name}")
                start = offset if mode == "ab" else 0
                print(f"GET {target.name}: {start:,}/{expected_size:,} bytes (attempt {attempt})", flush=True)
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        now = time.monotonic()
                        if now - last_report >= 30:
                            print(f"  {target.name}: {output.tell():,}/{expected_size:,} bytes", flush=True)
                            last_report = now
            actual_size = partial.stat().st_size
            if actual_size == expected_size:
                partial.replace(target)
                print(f"DONE {target.name}: {actual_size:,} bytes", flush=True)
                return
            if actual_size > expected_size:
                raise ValueError(f"Downloaded file exceeds remote size: {target.name}")
            print(f"Incomplete {target.name}: {actual_size:,}/{expected_size:,} bytes", flush=True)
        except (HTTPError, URLError, OSError) as exc:
            print(f"Retry {target.name}: {exc}", file=sys.stderr, flush=True)
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Could not complete {target.name}; rerun this command to resume {partial}")


def verify(dataset_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing sampling-2 files in {dataset_dir}: {', '.join(missing)}")
    for task in "ABCE":
        errors = check(task, dataset_dir)
        if errors:
            raise RuntimeError(f"Task {task} input check failed: {'; '.join(errors)}")
    print(f"PASS: Tasks A, B, C, E can read sampling-2 inputs at {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_v3",
                        help="Download destination; defaults to <repository>/dataset_v3")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="Sampling-2 directory URL; override for a mirror")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing inputs without network access or downloads")
    args = parser.parse_args()
    dataset_dir = (ROOT / args.dataset_dir).resolve() if not args.dataset_dir.is_absolute() else args.dataset_dir.resolve()
    if args.verify_only:
        verify(dataset_dir)
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/") + "/"
    files = {name: remote_size(base_url + name) for name in (*REQUIRED_FILES, *OPTIONAL_FILES)}
    bytes_needed = 0
    for name, size in files.items():
        target = dataset_dir / name
        if target.is_file() and target.stat().st_size == size:
            continue
        partial = target.with_name(target.name + ".part")
        resumed = partial.stat().st_size if partial.is_file() and partial.stat().st_size <= size else 0
        bytes_needed += size - resumed
    free_bytes = shutil.disk_usage(dataset_dir).free
    if bytes_needed > free_bytes:
        raise RuntimeError(
            f"Need {bytes_needed:,} free bytes for missing data in {dataset_dir}; "
            f"only {free_bytes:,} bytes are available."
        )
    for name, size in files.items():
        download_file(base_url + name, dataset_dir / name, size)
    verify(dataset_dir)
    print(f"Dataset ready at {dataset_dir}. Paper scripts use dataset_v3 by default.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
