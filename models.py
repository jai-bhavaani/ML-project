"""Factory for the six fraud classifiers used in the benchmark."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def build_models(y_train: np.ndarray, random_state: int = 42) -> dict[str, object]:
    """Create exactly six estimators configured for the training labels."""
    negatives = int(np.count_nonzero(y_train == 0))
    positives = int(np.count_nonzero(y_train == 1))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both 0 and 1 classes.")
    positive_weight = negatives / positives

    calibrated_svm = CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=random_state,
            ),
        ),
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )

    return {
        "SVM": calibrated_svm,
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            scale_pos_weight=positive_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            scale_pos_weight=positive_weight,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
        "TabNet": TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            seed=random_state,
            verbose=0,
        ),
    }