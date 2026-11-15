"""Create non-destructive, patient-aware stratified audio-data folds."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


LABELS = {"normal": 0, "abnormal": 1}


def patient_id(file_path: Path) -> str:
    return file_path.stem.split("_", maxsplit=1)[0]


def create_folds(source_dir: Path, output_dir: Path, n_splits: int, seed: int) -> None:
    records: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []
    for class_name, label in LABELS.items():
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for file_path in sorted(class_dir.glob("*.wav")):
            records.append(file_path)
            labels.append(label)
            groups.append(patient_id(file_path))

    if not records:
        raise ValueError(f"No WAV files found below: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.empty(len(records), dtype=np.int64)
    placeholder_x = np.zeros((len(records), 1), dtype=np.uint8)
    for fold_index, (_, test_indices) in enumerate(splitter.split(placeholder_x, labels, groups)):
        assignments[test_indices] = fold_index

    for fold_index in range(n_splits):
        for class_name in LABELS:
            (output_dir / f"fold_{fold_index + 1}" / class_name).mkdir(parents=True)

    counts: dict[tuple[int, str], int] = defaultdict(int)
    for file_path, label, fold_index in zip(records, labels, assignments, strict=True):
        class_name = next(name for name, value in LABELS.items() if value == label)
        shutil.copy2(file_path, output_dir / f"fold_{fold_index + 1}" / class_name / file_path.name)
        counts[(fold_index, class_name)] += 1

    for fold_index in range(n_splits):
        normal = counts[(fold_index, "normal")]
        abnormal = counts[(fold_index, "abnormal")]
        print(f"fold_{fold_index + 1}: {normal} normal, {abnormal} abnormal")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create patient-aware folds without changing source data.")
    parser.add_argument("--source-dir", type=Path, default=project_root / "ourdataset" / "ourdataset" / "train")
    parser.add_argument("--output-dir", type=Path, default=project_root / "ourdataset" / "five_folds")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    create_folds(arguments.source_dir, arguments.output_dir, arguments.n_splits, arguments.seed)
