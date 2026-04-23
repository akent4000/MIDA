"""Download + extract the RSNA Pneumonia Detection Challenge dataset.

The dataset (~3.6 GB compressed) lives on Kaggle and requires:
  1. A Kaggle account with the competition rules **accepted** via the web UI:
     https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
  2. `KAGGLE_API_TOKEN` set in the environment (new KGAT_* format — see
     backend/ml/README.md).

The script is idempotent: if the expected files are already under the target
directory, it does nothing. Use --force to wipe and re-fetch.

Usage:
    python backend/ml/training/fetch_rsna.py
    python backend/ml/training/fetch_rsna.py --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "rsna-pneumonia-detection-challenge"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "backend" / "ml" / "data" / "rsna"

# Files / dirs we expect after extraction — used both as a "done" sentinel
# and as a sanity check that the archive structure hasn't changed upstream.
EXPECTED_ARTIFACTS = [
    "stage_2_train_labels.csv",
    "stage_2_detailed_class_info.csv",
    "stage_2_train_images",
    "stage_2_test_images",
]


def already_downloaded(target: Path) -> bool:
    return all((target / name).exists() for name in EXPECTED_ARTIFACTS)


def credentials_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    return legacy.exists()


def run_kaggle_download(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(target),
    ]
    print(f"[fetch_rsna] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def extract_zips(target: Path) -> None:
    # Kaggle ships the competition as a single outer zip; some competitions
    # additionally nest inner zips (e.g. one per image folder). Unpack any
    # zip we find, repeatedly, until there are none left to process.
    while True:
        zips = sorted(target.rglob("*.zip"))
        if not zips:
            return
        for zip_path in zips:
            print(f"[fetch_rsna] extracting {zip_path.relative_to(target)}", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(zip_path.parent)
            zip_path.unlink()


def summarize(target: Path) -> None:
    dicoms = sum(1 for _ in target.rglob("*.dcm"))
    csvs = sorted(p.name for p in target.glob("*.csv"))
    missing = [name for name in EXPECTED_ARTIFACTS if not (target / name).exists()]
    print(f"[fetch_rsna] {dicoms} DICOM files under {target}")
    print(f"[fetch_rsna] CSVs: {csvs}")
    if missing:
        print(f"[fetch_rsna] WARNING: missing expected artefacts: {missing}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe target dir and redownload even if data already present.",
    )
    args = parser.parse_args()

    if not credentials_present():
        print(
            "ERROR: no Kaggle credentials found.\n"
            "       Set KAGGLE_API_TOKEN (see backend/ml/README.md).",
            file=sys.stderr,
        )
        return 1

    if already_downloaded(DATA_DIR) and not args.force:
        print(f"[fetch_rsna] dataset already present at {DATA_DIR} — skipping")
        print("[fetch_rsna] re-run with --force to refetch")
        summarize(DATA_DIR)
        return 0

    if args.force and DATA_DIR.exists():
        print(f"[fetch_rsna] --force: removing {DATA_DIR}")
        shutil.rmtree(DATA_DIR)

    run_kaggle_download(DATA_DIR)
    extract_zips(DATA_DIR)
    summarize(DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
