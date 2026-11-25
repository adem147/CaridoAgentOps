import argparse
import os
from pathlib import Path

import numpy as np
import librosa
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# Import from the models directory
from models.decision_tree import AudioDecisionTree
from preprocessing import resample_audio
from features import extract_features

# Pathological class is assumed to be 1
PATHOLOGICAL_CLASS = 1
NORMAL_CLASS = 0

def save_model_to_mlflow(pathological_recall, normal_recall, accuracy, n_folds, model_name="DecisionTree"):
    # SQLite is supported by current MLflow releases and works locally/Colab.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    # Create or select an experiment
    mlflow.set_experiment("CardioAgent_Experiments")

    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("cross_validation_folds", n_folds)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("pathological_recall", pathological_recall)
        mlflow.log_metric("normal_recall", normal_recall)

    print("Experiment logged successfully!")

CLASS_LABELS = {"normal": NORMAL_CLASS, "abnormal": PATHOLOGICAL_CLASS}
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "ourdataset" / "ourdataset"
DEFAULT_FOLDS_DIR = Path(__file__).resolve().parents[2] / "ourdataset" / "five_folds"


def load_real_dataset(base_path: str | Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Load labelled WAV files and turn each recording into one feature vector.

    Expected directory layout:
        <base_path>/<split>/normal/*.wav
        <base_path>/<split>/abnormal/*.wav

    Files that cannot be decoded or transformed are skipped and reported after
    loading.  Both classes must contain at least one usable recording.
    """
    dataset_dir = Path(base_path).expanduser().resolve()
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory was not found: {split_dir}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    loaded_counts = {category: 0 for category in CLASS_LABELS}
    skipped_files: list[tuple[Path, str]] = []

    print(f"Loading dataset from: {split_dir}")
    for category, label in CLASS_LABELS.items():
        class_dir = split_dir / category
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory was not found: {class_dir}")

        wav_files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() == ".wav")
        print(f"Processing {len(wav_files)} recordings from {category}...")

        for file_path in wav_files:
            try:
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
                if audio.size == 0:
                    raise ValueError("audio contains no samples")

                resampled_audio = resample_audio(audio, sample_rate, target_sr=2000)
                features.append(extract_features(resampled_audio, sr=2000))
                labels.append(label)
                loaded_counts[category] += 1
            except Exception as error:
                skipped_files.append((file_path, str(error)))

    missing_classes = [category for category, count in loaded_counts.items() if count == 0]
    if missing_classes:
        raise ValueError(f"No usable recordings were loaded for: {', '.join(missing_classes)}")

    print(
        "Dataset ready: "
        f"{loaded_counts['normal']} normal, "
        f"{loaded_counts['abnormal']} abnormal, "
        f"{len(skipped_files)} skipped."
    )
    if skipped_files:
        print("First skipped files:")
        for file_path, reason in skipped_files[:5]:
            print(f"  - {file_path.name}: {reason}")

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def get_fold_directories(folds_dir: str | Path) -> list[Path]:
    """Return the existing fold directories in a deterministic order."""
    folds_path = Path(folds_dir).expanduser().resolve()
    fold_dirs = sorted(path for path in folds_path.glob("fold_*") if path.is_dir())
    if len(fold_dirs) < 2:
        raise ValueError(f"Expected at least two fold directories below: {folds_path}")
    return fold_dirs


def train_only_decision_tree(folds_dir: str | Path) -> AudioDecisionTree:
    """Evaluate with the pre-created folds, then train one final model on all data."""
    fold_dirs = get_fold_directories(folds_dir)
    print(f"--- DecisionTree {len(fold_dirs)}-Fold Cross-Validation ---")

    # Load each physical fold only once.  Every model run below uses these exact
    # partitions, making results comparable to the CNN's cross-validation run.
    datasets = [
        (fold_dir.name, *load_real_dataset(folds_dir, split=fold_dir.name))
        for fold_dir in fold_dirs
    ]

    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    for test_index, (fold_name, X_test, y_test) in enumerate(datasets):
        X_train = np.concatenate([X for index, (_, X, _) in enumerate(datasets) if index != test_index])
        y_train = np.concatenate([y for index, (_, _, y) in enumerate(datasets) if index != test_index])

        model = AudioDecisionTree(max_depth=10)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(np.mean(predictions == y_test)),
            "pathological_recall": float(recall_score(y_test, predictions, pos_label=PATHOLOGICAL_CLASS, zero_division=0)),
            "normal_recall": float(recall_score(y_test, predictions, pos_label=NORMAL_CLASS, zero_division=0)),
        }
        fold_metrics.append(metrics)
        all_predictions.append(predictions)
        all_targets.append(y_test)
        print(
            f"{fold_name}: accuracy={metrics['accuracy']:.4f}, "
            f"pathological_recall={metrics['pathological_recall']:.4f}, "
            f"normal_recall={metrics['normal_recall']:.4f}"
        )

    aggregate_metrics = {
        metric: float(np.mean([result[metric] for result in fold_metrics]))
        for metric in fold_metrics[0]
    }
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    print("\n--- Mean cross-validation metrics ---")
    for metric, value in aggregate_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nClassification Report (all held-out predictions):")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix (all held-out predictions):")
    print(confusion_matrix(y_true, y_pred))

    try:
        save_model_to_mlflow(**aggregate_metrics, n_folds=len(fold_dirs), model_name="DecisionTree")
    except Exception as error:
        print(f"MLflow logging skipped: {error}")

    # Cross-validation models are only for evaluation.  This is the single
    # final tree to export or deploy after comparison with other models.
    final_model = AudioDecisionTree(max_depth=10)
    final_model.train(
        np.concatenate([X for _, X, _ in datasets]),
        np.concatenate([y for _, _, y in datasets]),
    )
    return final_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Decision Tree with pre-created validation folds.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_FOLDS_DIR,
        help="Directory containing fold_1 through fold_N directories.",
    )
    args = parser.parse_args()

    try:
        trained_model = train_only_decision_tree(args.dataset_dir)
        print("\nCross-validation and final training successful.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
