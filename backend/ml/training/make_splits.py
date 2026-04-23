"""Build reproducible train/val/test splits for RSNA Pneumonia.

The RSNA label CSV has one row per bounding box, so a single patient can appear
multiple times. Splitting by row would leak patients across splits — we split
by patientId instead.

Stratification key (composite):
    detailed_class x PatientSex x ViewPosition

Dropping `ViewPosition`/`PatientSex` from the key (via --no-dicom-meta) falls
back to stratifying on `detailed_class` alone, which is enough to keep the
positive rate balanced but loses the AP/PA and sex balance.

Target proportions: 70 / 15 / 15 (train / val / test). Seed is fixed so the
output JSON is byte-identical across runs and reviewable in git.

Output JSON shape:
    {
        "meta": {...reproducibility info...},
        "train": ["<patientId>", ...],
        "val":   ["<patientId>", ...],
        "test":  ["<patientId>", ...]
    }

Usage:
    python backend/ml/training/make_splits.py
    python backend/ml/training/make_splits.py --no-dicom-meta
    python backend/ml/training/make_splits.py --output splits_rsna_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "backend" / "ml" / "data" / "rsna"
CONFIG_DIR = REPO_ROOT / "backend" / "ml" / "configs"

TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
CLASS_INFO_CSV = DATA_DIR / "stage_2_detailed_class_info.csv"

DEFAULT_OUTPUT = "splits_rsna_v1.json"
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Minimum members per stratum needed for sklearn to split it — anything below
# this gets diverted straight into train to avoid ValueErrors.
MIN_STRATUM = 2


def build_patient_frame(include_dicom_meta: bool) -> pd.DataFrame:
    """One row per patientId with Target, detailed_class, sex, view."""
    labels = pd.read_csv(LABELS_CSV)
    class_info = pd.read_csv(CLASS_INFO_CSV)

    per_patient = labels.groupby("patientId")["Target"].max().reset_index()
    class_per_patient = class_info.drop_duplicates("patientId")[["patientId", "class"]]
    df = per_patient.merge(class_per_patient, on="patientId", how="left").rename(
        columns={"class": "detailed_class"}
    )

    if include_dicom_meta:
        print(f"[make_splits] reading DICOM metadata for {len(df)} patients...", flush=True)
        sexes: list[str] = []
        views: list[str] = []
        for pid in tqdm(df["patientId"], ncols=80):
            ds = pydicom.dcmread(TRAIN_IMG_DIR / f"{pid}.dcm", stop_before_pixels=True)
            sexes.append(str(ds.get("PatientSex", "?")))
            views.append(str(ds.get("ViewPosition", "?")))
        df["sex"] = sexes
        df["view"] = views
        df["strat_key"] = df["detailed_class"] + "|" + df["sex"] + "|" + df["view"]
    else:
        df["sex"] = "?"
        df["view"] = "?"
        df["strat_key"] = df["detailed_class"]

    return df


def _filter_rare(df: pd.DataFrame, col: str, min_count: int = MIN_STRATUM):
    counts = df[col].value_counts()
    rare = set(counts[counts < min_count].index)
    return df[~df[col].isin(rare)].copy(), df[df[col].isin(rare)].copy()


def stratified_3way_split(
    df: pd.DataFrame,
    seed: int = SEED,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout_frac = val_frac + test_frac

    df_strat, df_rare_1 = _filter_rare(df, "strat_key")
    train_df, holdout_df = train_test_split(
        df_strat,
        test_size=holdout_frac,
        stratify=df_strat["strat_key"],
        random_state=seed,
    )

    holdout_strat, holdout_rare_2 = _filter_rare(holdout_df, "strat_key")
    val_df, test_df = train_test_split(
        holdout_strat,
        test_size=test_frac / holdout_frac,
        stratify=holdout_strat["strat_key"],
        random_state=seed,
    )

    if len(df_rare_1) or len(holdout_rare_2):
        print(
            f"[make_splits] {len(df_rare_1) + len(holdout_rare_2)} "
            "rare-stratum rows diverted to train",
            flush=True,
        )
        train_df = pd.concat([train_df, df_rare_1, holdout_rare_2], ignore_index=True)

    return train_df, val_df, test_df


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_set = set(train_df["patientId"])
    val_set = set(val_df["patientId"])
    test_set = set(test_df["patientId"])
    assert not (train_set & val_set), "train/val overlap"
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set & test_set), "val/test overlap"
    print(
        f"[make_splits] no overlaps; "
        f"{len(train_set) + len(val_set) + len(test_set)} patients total"
    )


def print_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n[make_splits] target distribution (fraction positive):")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s}: n={len(d):6d}  pos_rate={d['Target'].mean():.3f}")

    print("\n[make_splits] detailed_class distribution (%):")
    table = (
        pd.DataFrame(
            {
                "train": train_df["detailed_class"].value_counts(normalize=True),
                "val": val_df["detailed_class"].value_counts(normalize=True),
                "test": test_df["detailed_class"].value_counts(normalize=True),
            }
        ).fillna(0)
        * 100
    )
    print(table.to_string(float_format="%5.2f"))

    if "sex" in train_df.columns and (train_df["sex"] != "?").any():
        print("\n[make_splits] sex distribution (%):")
        sex_tbl = (
            pd.DataFrame(
                {
                    "train": train_df["sex"].value_counts(normalize=True),
                    "val": val_df["sex"].value_counts(normalize=True),
                    "test": test_df["sex"].value_counts(normalize=True),
                }
            ).fillna(0)
            * 100
        )
        print(sex_tbl.to_string(float_format="%5.2f"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output filename (written under backend/ml/configs/).",
    )
    parser.add_argument(
        "--no-dicom-meta",
        action="store_true",
        help="Skip DICOM metadata read; stratify on detailed_class only (fast).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not LABELS_CSV.exists():
        print(
            f"ERROR: {LABELS_CSV} not found. Run fetch_rsna.py first.",
            file=sys.stderr,
        )
        return 1

    df = build_patient_frame(include_dicom_meta=not args.no_dicom_meta)
    train_df, val_df, test_df = stratified_3way_split(df, seed=args.seed)
    validate_splits(train_df, val_df, test_df)
    print_distribution(train_df, val_df, test_df)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG_DIR / args.output
    payload = {
        "meta": {
            "seed": args.seed,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "dicom_meta_used": not args.no_dicom_meta,
            "stratification_key": (
                "detailed_class|sex|view" if not args.no_dicom_meta else "detailed_class"
            ),
            "train_n": len(train_df),
            "val_n": len(val_df),
            "test_n": len(test_df),
        },
        "train": sorted(train_df["patientId"].tolist()),
        "val": sorted(val_df["patientId"].tolist()),
        "test": sorted(test_df["patientId"].tolist()),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[make_splits] written: {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
