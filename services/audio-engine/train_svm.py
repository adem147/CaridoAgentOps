import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

from models.svm import AudioSVM
from train_decision_tree import (
    NORMAL_CLASS,
    PATHOLOGICAL_CLASS,
    DEFAULT_FOLDS_DIR,
    get_fold_directories,
    load_real_dataset,
)


def save_model_to_mlflow(pathological_recall, normal_recall, accuracy, n_folds, C):
    """Log aggregate cross-validation metrics locally unless a server is configured."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("CardioAgent_Experiments")

    with mlflow.start_run():
        mlflow.log_param("model", "SVM")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", C)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("cross_validation_folds", n_folds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("pathological_recall", pathological_recall)
        mlflow.log_metric("normal_recall", normal_recall)


def train_only_svm(folds_dir: str | Path, C: float = 1.0) -> AudioSVM:
    """Evaluate an RBF SVM on fixed folds, then fit one final model on all data."""
    fold_dirs = get_fold_directories(folds_dir)
    print(f"--- SVM {len(fold_dirs)}-Fold Cross-Validation ---")

    # The same 149 features and named data partitions used by the decision tree.
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

        # SVMs are sensitive to feature scale.  The scaler learns only from
        # training folds, then transforms the held-out fold without leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = AudioSVM(kernel="rbf", C=C, class_weight="balanced")
        model.train(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

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
    try:
        save_model_to_mlflow(**aggregate_metrics, n_folds=len(fold_dirs), C=C)
    except Exception as error:
        # Tracking must not prevent the evaluation report or final model from
        # being produced when MLflow is unavailable or misconfigured.
        print(f"MLflow logging skipped: {error}")

    print("\n--- Mean cross-validation metrics ---")
    for metric, value in aggregate_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nClassification Report (all held-out predictions):")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix (all held-out predictions):")
    print(confusion_matrix(y_true, y_pred))

    final_X = np.concatenate([X for _, X, _ in datasets])
    final_y = np.concatenate([y for _, _, y in datasets])
    final_scaler = StandardScaler()
    final_model = AudioSVM(kernel="rbf", C=C, class_weight="balanced")
    final_model.train(final_scaler.fit_transform(final_X), final_y)
    final_model.scaler = final_scaler
    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SVM with pre-created validation folds.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter.")
    args = parser.parse_args()

    try:
        trained_model = train_only_svm(args.dataset_dir, C=args.C)
        print("\nCross-validation and final training successful.")
    except Exception as error:
        print(f"\nAn error occurred during training: {error}")
