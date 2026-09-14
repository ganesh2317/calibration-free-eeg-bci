"""PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB) Downloader.

Downloads genuine raw EDF files from PhysioNet concurrently with retries:
https://physionet.org/files/eegmmidb/1.0.0/
"""
import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import CONFIG


def download_single_file(task: Tuple[str, Path, str, str]) -> Tuple[str, str, bool, str]:
    """Download a single EDF file task with retries.

    Args:
        task: Tuple of (url, dest_path, subject, run).

    Returns:
        Tuple of (subject, run, success, message).
    """
    url, dest_path, subject, run = task
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If already downloaded and valid (> 1MB), skip
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        return subject, run, True, "Already exists"

    temp_dest = dest_path.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EEG-BCI/1.0"}

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as response, open(temp_dest, "wb") as out_file:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if temp_dest.exists() and temp_dest.stat().st_size > 1024 * 1024:
                if dest_path.exists():
                    dest_path.unlink()
                temp_dest.rename(dest_path)
                return subject, run, True, f"Success ({dest_path.stat().st_size / (1024*1024):.2f} MB)"
            else:
                if temp_dest.exists():
                    temp_dest.unlink()
        except Exception as e:
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(1.0 * attempt)
            else:
                return subject, run, False, f"Failed after {max_retries} attempts: {e}"

    return subject, run, False, "Download error"


def download_cohort(
    subjects: Optional[List[str]] = None,
    runs: Optional[List[str]] = None,
    max_workers: int = 8,
) -> dict:
    """Download dataset for cohort of subjects concurrently.

    Args:
        subjects: List of subject IDs.
        runs: List of runs per subject.
        max_workers: Concurrency level for downloads.

    Returns:
        Dictionary mapping subject ID to list of downloaded EDF file paths.
    """
    subjects = subjects or CONFIG.dataset.default_subjects
    runs = runs or CONFIG.dataset.runs
    base_url = CONFIG.dataset.base_url.rstrip("/")
    raw_dir = CONFIG.paths.raw_data_dir

    print(f"=== Starting Concurrent Download of PhysioNet EEGMMIDB ===")
    print(f"Subjects ({len(subjects)}): {', '.join(subjects)}")
    print(f"Runs ({len(runs)}): {', '.join(runs)}")
    print(f"Total files to retrieve: {len(subjects) * len(runs)}")
    print(f"Concurrency threads: {max_workers}")
    print(f"Target directory: {raw_dir}")
    print("=" * 60)

    tasks = []
    for subj in subjects:
        subj_dir = raw_dir / subj
        for run in runs:
            filename = f"{subj}{run}.edf"
            url = f"{base_url}/{subj}/{filename}"
            dest = subj_dir / filename
            tasks.append((url, dest, subj, run))

    results = {s: [] for s in subjects}
    completed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_file, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Total Progress", unit="file") as pbar:
            for future in as_completed(futures):
                subj, run, success, msg = future.result()
                pbar.update(1)
                if success:
                    completed_count += 1
                    dest_path = raw_dir / subj / f"{subj}{run}.edf"
                    results[subj].append(dest_path)
                else:
                    failed_count += 1
                    print(f"\n[FAIL] {subj}{run}: {msg}")

    print(f"\n=== Download Finished: {completed_count}/{len(tasks)} files ready, {failed_count} failures ===")
    if failed_count > 0:
        raise RuntimeError(f"Download incomplete: {failed_count} files failed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhysioNet EEG Motor Movement/Imagery Dataset")
    parser.add_argument("--subjects", nargs="+", default=CONFIG.dataset.default_subjects, help="Subject IDs (e.g. S001 S002)")
    parser.add_argument("--runs", nargs="+", default=CONFIG.dataset.runs, help="Runs (e.g. R04 R05 R06 R08 R09 R10)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    args = parser.parse_args()

    download_cohort(subjects=args.subjects, runs=args.runs, max_workers=args.workers)
