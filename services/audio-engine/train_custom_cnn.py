import argparse
import os
from pathlib import Path

import librosa
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from features import extract_features
from models.custom_cnn import AudioCustomCNN
from preprocessing import resample_audio


PATHOLOGICAL_CLASS = 1
NORMAL_CLASS = 0
CLASS_LABELS = {"normal": NORMAL_CLASS, "abnormal": PATHOLOGICAL_CLASS}
DEFAULT_FOLDS_DIR = Path(__file__).resolve().parents[2] / "ourdataset" / "five_folds"


def save_model_to_mlflow(pathological_recall, normal_recall, accuracy, n_folds, model_name="CustomCNN"):
    # Colab has no MLflow server by default, so use local files unless a server
    # URI is explicitly supplied through MLFLOW_TRACKING_URI.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("CardioAgent_Experiments")

    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("cross_validation_folds", n_folds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("pathological_recall", pathological_recall)
        mlflow.log_metric("normal_recall", normal_recall)

    print("Experiment logged successfully!")


def load_real_dataset(base_path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one physical fold using the CNN's MFCC-only feature representation."""
    split_dir = Path(base_path).expanduser().resolve() / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset fold directory was not found: {split_dir}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    loaded_counts = {category: 0 for category in CLASS_LABELS}
    skipped_files: list[tuple[Path, str]] = []

    print(f"Loading dataset from: {split_dir}")
    for category, label in CLASS_LABELS.items():
        class_dir = split_dir / category
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory was not found: {class_dir}")

        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"Processing {len(wav_files)} recordings from {category}...")
        for file_path in wav_files:
            try:
                audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
                if audio.size == 0:
                    raise ValueError("audio contains no samples")

                resampled_audio = resample_audio(audio, sample_rate, target_sr=2000)
                features.append(extract_features(resampled_audio, sr=2000, mode="mfcc"))
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
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def get_fold_directories(folds_dir: str | Path) -> list[Path]:
    folds_path = Path(folds_dir).expanduser().resolve()
    fold_dirs = sorted(path for path in folds_path.glob("fold_*") if path.is_dir())
    if len(fold_dirs) < 2:
        raise ValueError(f"Expected at least two fold directories below: {folds_path}")
    return fold_dirs


def train_only_custom_cnn(folds_dir: str | Path) -> AudioCustomCNN:
    """Evaluate on fixed folds, then fit one final CNN on all recordings."""
    fold_dirs = get_fold_directories(folds_dir)
    print(f"--- CustomCNN {len(fold_dirs)}-Fold Cross-Validation ---")

    # These named physical partitions are also used by the decision tree,
    # keeping its train/test recordings identical to this model's.
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

        model = AudioCustomCNN(input_dim=X_train.shape[1])
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
    save_model_to_mlflow(**aggregate_metrics, n_folds=len(fold_dirs), model_name="CustomCNN")

    print("\n--- Mean cross-validation metrics ---")
    for metric, value in aggregate_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nClassification Report (all held-out predictions):")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix (all held-out predictions):")
    print(confusion_matrix(y_true, y_pred))

    final_model = AudioCustomCNN(input_dim=datasets[0][1].shape[1])
    final_model.train(
        np.concatenate([X for _, X, _ in datasets]),
        np.concatenate([y for _, _, y in datasets]),
    )
    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Custom CNN with pre-created validation folds.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_FOLDS_DIR,
        help="Directory containing fold_1 through fold_N directories.",
    )
    args = parser.parse_args()

    try:
        trained_model = train_only_custom_cnn(args.dataset_dir)
        print("\nCross-validation and final training successful.")
    except Exception as error:
        print(f"\nAn error occurred during training: {error}")
