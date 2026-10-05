"""Train all six classifiers on each dataset and write benchmark results."""

from __future__ import annotations

import argparse
import re
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models import build_models
from preprocess import load_and_split_dataset
from visualize import generate_reports


DATASETS = ("creditcard", "paysim", "ieee_cis")


def _fit_model(model: object, X_train: np.ndarray, y_train: np.ndarray,
               tabnet_epochs: int, random_state: int) -> None:
    if isinstance(model, TabNetClassifier):
        fit_X, valid_X, fit_y, valid_y = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=random_state,
        )
        model.fit(
            fit_X,
            fit_y,
            eval_set=[(valid_X, valid_y)],
            eval_name=["validation"],
            eval_metric=["auc"],
            max_epochs=tabnet_epochs,
            patience=min(10, tabnet_epochs),
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            weights=1,
            drop_last=False,
        )
    else:
        model.fit(X_train, y_train)


def _probabilities(model: object, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Classifier did not return probabilities for both classes.")
    return probabilities[:, 1].astype(np.float64, copy=False)


def _artifact_name(dataset: str, model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{dataset}_{slug}"


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    test_labels: dict[str, np.ndarray] = {}

    for dataset_name in DATASETS if args.dataset == "all" else (args.dataset,):
        print(f"\nPreparing {dataset_name} ...", flush=True)
        split = load_and_split_dataset(
            dataset_name,
            data_dir=args.data_dir,
            random_state=args.random_state,
            paysim_sample_size=args.paysim_sample_size,
            ieee_sample_size=args.ieee_sample_size,
        )
        print(
            f"  train={split.X_train.shape}, test={split.X_test.shape}, "
            f"fraud train={int(split.y_train.sum())}, test={int(split.y_test.sum())}",
            flush=True,
        )
        dataset_probabilities: dict[str, np.ndarray] = {}
        test_labels[dataset_name] = split.y_test
        models = build_models(split.y_train, random_state=args.random_state)

        for model_name, model in models.items():
            print(f"  Training {model_name} ...", flush=True)
            started = time.perf_counter()
            try:
                _fit_model(
                    model,
                    split.X_train,
                    split.y_train,
                    args.tabnet_epochs,
                    args.random_state,
                )
                training_seconds = time.perf_counter() - started
                scores = _probabilities(model, split.X_test)
                predicted = (scores >= 0.5).astype(np.int8)
                record: dict[str, object] = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "pr_auc": average_precision_score(split.y_test, scores),
                    "roc_auc": roc_auc_score(split.y_test, scores),
                    "precision": precision_score(split.y_test, predicted, zero_division=0),
                    "recall": recall_score(split.y_test, predicted, zero_division=0),
                    "f1_score": f1_score(split.y_test, predicted, zero_division=0),
                    "training_seconds": training_seconds,
                    "status": "ok",
                    "error": "",
                }
                dataset_probabilities[model_name] = scores
                artifact = model_dir / _artifact_name(dataset_name, model_name)
                if isinstance(model, TabNetClassifier):
                    model.save_model(str(artifact))
                else:
                    joblib.dump(model, artifact.with_suffix(".joblib"), compress=3)
                print(
                    f"    PR-AUC={record['pr_auc']:.5f} ROC-AUC={record['roc_auc']:.5f} "
                    f"fit={training_seconds:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                training_seconds = time.perf_counter() - started
                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "training_seconds": training_seconds,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"    FAILED: {record['error']}", flush=True)
                if args.verbose_errors:
                    traceback.print_exc()
            records.append(record)
            del model

        probabilities[dataset_name] = dataset_probabilities
        del split, models

    results = pd.DataFrame.from_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "overall_benchmark_results.csv", index=False)
    generate_reports(results, probabilities, test_labels, output_dir)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".", help="Directory containing the dataset CSV files.")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics, charts, and models.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--paysim-sample-size", type=int, default=100_000)
    parser.add_argument("--ieee-sample-size", type=int, default=50_000)
    parser.add_argument("--tabnet-epochs", type=int, default=50)
    parser.add_argument("--verbose-errors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark = run_benchmark(parse_args())
    print("\nBenchmark complete:")
    print(benchmark[["dataset", "model", "pr_auc", "roc_auc", "training_seconds", "status"]].to_string(index=False))