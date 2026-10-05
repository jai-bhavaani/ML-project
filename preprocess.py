"""Dataset discovery and leakage-safe preprocessing for fraud benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer


@dataclass
class DatasetSplit:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def locate_dataset_files(data_dir: str | Path) -> dict[str, Path]:
    """Find the three datasets using their conventional filenames."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    paysim_candidates = sorted(root.glob("PS_*.csv")) + sorted(root.glob("paysim.csv"))
    files = {
        "creditcard": root / "creditcard.csv",
        "ieee_cis_transaction": root / "train_transaction.csv",
        "ieee_cis_identity": root / "train_identity.csv",
    }
    missing = [str(path.name) for path in files.values() if not path.is_file()]
    if not paysim_candidates:
        missing.append("PS_*.csv or paysim.csv")
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset file(s) in {root}: {', '.join(missing)}"
        )
    files["paysim"] = paysim_candidates[0]
    return files


def _stratified_indices(
    labels: np.ndarray, sample_size: int, random_state: int
) -> np.ndarray:
    indices = np.arange(len(labels), dtype=np.int64)
    if sample_size >= len(indices):
        return indices
    if sample_size < 2 or np.unique(labels).size < 2:
        raise ValueError("A stratified sample needs at least two rows and two classes.")
    selected, _ = train_test_split(
        indices,
        train_size=sample_size,
        stratify=labels,
        random_state=random_state,
    )
    return np.sort(selected)


def _fit_encode_split(
    frame: pd.DataFrame,
    target: str,
    name: str,
    random_state: int,
    test_size: float,
) -> DatasetSplit:
    if target not in frame.columns:
        raise ValueError(f"Dataset {name!r} is missing target column {target!r}.")
    labels = frame[target].astype("int8").to_numpy()
    features = frame.drop(columns=[target]).copy()
    features = features.loc[:, features.notna().any(axis=0)]

    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    train = features.iloc[train_idx]
    test = features.iloc[test_idx]
    numeric_columns = train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    arrays: list[np.ndarray] = []
    feature_names: list[str] = []

    if numeric_columns:
        imputer = SimpleImputer(strategy="median")
        train_numeric = imputer.fit_transform(train[numeric_columns]).astype(np.float32)
        test_numeric = imputer.transform(test[numeric_columns]).astype(np.float32)
        if name == "creditcard":
            scaled_columns = [column for column in ("Time", "Amount") if column in numeric_columns]
            if scaled_columns:
                scaler = RobustScaler()
                positions = [numeric_columns.index(column) for column in scaled_columns]
                train_numeric[:, positions] = scaler.fit_transform(train_numeric[:, positions])
                test_numeric[:, positions] = scaler.transform(test_numeric[:, positions])
        arrays.append(train_numeric)
        feature_names.extend(numeric_columns)

    if categorical_columns:
        train_categorical = train[categorical_columns].astype("string").fillna("__MISSING__")
        test_categorical = test[categorical_columns].astype("string").fillna("__MISSING__")
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
        train_encoded = encoder.fit_transform(train_categorical)
        test_encoded = encoder.transform(test_categorical)
        arrays.append(train_encoded)
        feature_names.extend(categorical_columns)

    if not arrays:
        raise ValueError(f"Dataset {name!r} has no usable feature columns.")

    X_train = np.concatenate(arrays, axis=1, dtype=np.float32)
    test_arrays: list[np.ndarray] = []
    if numeric_columns:
        test_arrays.append(test_numeric)
    if categorical_columns:
        test_arrays.append(test_encoded)
    X_test = np.concatenate(test_arrays, axis=1, dtype=np.float32)
    return DatasetSplit(
        name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=labels[train_idx],
        y_test=labels[test_idx],
        feature_names=feature_names,
    )


def _load_creditcard(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Class" not in frame.columns:
        raise ValueError(f"{path.name} does not contain the required 'Class' target.")
    return frame


def _load_paysim(path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    eligible_indices: list[np.ndarray] = []
    eligible_labels: list[np.ndarray] = []
    offset = 0
    for chunk in pd.read_csv(path, usecols=["type", "isFraud"], chunksize=250_000):
        mask = chunk["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()
        local_indices = np.flatnonzero(mask) + offset
        if len(local_indices):
            eligible_indices.append(local_indices)
            eligible_labels.append(chunk.loc[mask, "isFraud"].to_numpy(dtype=np.int8))
        offset += len(chunk)

    if not eligible_indices:
        raise ValueError(f"No TRANSFER or CASH_OUT rows were found in {path.name}.")
    candidates = np.concatenate(eligible_indices)
    labels = np.concatenate(eligible_labels)
    selected = _stratified_indices(labels, sample_size, random_state)
    selected_rows = np.sort(candidates[selected])
    pieces: list[pd.DataFrame] = []
    offset = 0
    selected_cursor = 0
    for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
        chunk_end = offset + len(chunk)
        start_cursor = selected_cursor
        while selected_cursor < len(selected_rows) and selected_rows[selected_cursor] < chunk_end:
            selected_cursor += 1
        if selected_cursor > start_cursor:
            local = selected_rows[start_cursor:selected_cursor] - offset
            pieces.append(chunk.iloc[local])
        offset = chunk_end
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.loc[frame["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    frame["error_orig"] = frame["newbalanceOrig"] + frame["amount"] - frame["oldbalanceOrg"]
    frame["error_dest"] = frame["oldbalanceDest"] + frame["amount"] - frame["newbalanceDest"]
    frame.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore", inplace=True)
    return frame


def _load_ieee(
    transaction_path: Path,
    identity_path: Path,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    labels = pd.read_csv(transaction_path, usecols=["isFraud"])["isFraud"].to_numpy(dtype=np.int8)
    selected_rows = _stratified_indices(labels, sample_size, random_state)
    identity = pd.read_csv(identity_path, low_memory=False).drop_duplicates("TransactionID")

    selected_pieces: list[pd.DataFrame] = []
    non_null_counts: pd.Series | None = None
    offset = 0
    cursor = 0
    row_count = 0
    for chunk in pd.read_csv(transaction_path, chunksize=75_000, low_memory=False):
        counts = chunk.notna().sum()
        non_null_counts = counts if non_null_counts is None else non_null_counts.add(counts, fill_value=0)
        end = offset + len(chunk)
        start_cursor = cursor
        while cursor < len(selected_rows) and selected_rows[cursor] < end:
            cursor += 1
        if cursor > start_cursor:
            selected_pieces.append(chunk.iloc[selected_rows[start_cursor:cursor] - offset])
        offset = end
        row_count += len(chunk)

    transaction_sample = pd.concat(selected_pieces, ignore_index=True)
    identity_features = [column for column in identity.columns if column != "TransactionID"]
    identity_counts = identity[identity_features].notna().sum()
    transaction_counts = non_null_counts
    combined_counts = pd.concat([transaction_counts, identity_counts])
    keep_columns = combined_counts[combined_counts >= 0.25 * row_count].index.tolist()
    transaction_sample = transaction_sample.drop(
        columns=[column for column in transaction_sample.columns if column not in keep_columns],
        errors="ignore",
    )
    identity = identity[["TransactionID"] + [column for column in identity_features if column in keep_columns]]
    merged = transaction_sample.merge(identity, on="TransactionID", how="left", sort=False)
    merged.drop(columns=["TransactionID", "TransactionDT"], errors="ignore", inplace=True)
    return merged


def load_and_split_dataset(
    dataset: str,
    data_dir: str | Path = ".",
    random_state: int = 42,
    test_size: float = 0.2,
    paysim_sample_size: int = 100_000,
    ieee_sample_size: int = 50_000,
) -> DatasetSplit:
    """Load, preprocess, and stratify-split one supported dataset."""
    files = locate_dataset_files(data_dir)
    if dataset == "creditcard":
        frame = _load_creditcard(files["creditcard"])
        target = "Class"
    elif dataset == "paysim":
        frame = _load_paysim(files["paysim"], paysim_sample_size, random_state)
        target = "isFraud"
    elif dataset == "ieee_cis":
        frame = _load_ieee(
            files["ieee_cis_transaction"],
            files["ieee_cis_identity"],
            ieee_sample_size,
            random_state,
        )
        target = "isFraud"
    else:
        raise ValueError("dataset must be one of: creditcard, paysim, ieee_cis")
    return _fit_encode_split(frame, target, dataset, random_state, test_size)