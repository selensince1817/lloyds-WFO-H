"""Data loading and feature-preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    COLUMN_RENAMES,
    DROP_COLUMNS,
    NUMERIC_COLUMNS,
    PROCESSED_EXPORT_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the original Lloyds dataset."""
    dataset_path = Path(path) if path else RAW_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return pd.read_csv(dataset_path)


def prepare_feature_frame(
    df: pd.DataFrame,
    *,
    expect_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Clean column names, enforce schema, and split features/target.

    Parameters
    ----------
    df:
        Raw dataframe matching the original questionnaire schema.
    expect_target:
        Set False when preparing rows for inference (no `Target` column).
    """

    working_df = df.copy()
    working_df.columns = [col.strip() for col in working_df.columns]
    working_df.rename(columns=COLUMN_RENAMES, inplace=True)
    working_df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

    missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in working_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_cols))
            + f". Expected columns: {', '.join(ALL_FEATURE_COLUMNS)}"
        )

    features = working_df[ALL_FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        features[column] = (
            features[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
            .replace({"": pd.NA, "nan": pd.NA})
        )

    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    target = None
    if expect_target:
        if TARGET_COLUMN not in working_df.columns:
            raise ValueError(
                f"Target column `{TARGET_COLUMN}` is missing. "
                "Set expect_target=False if you intend to run inference."
            )
        target_series = pd.to_numeric(working_df[TARGET_COLUMN], errors="coerce")
        if target_series.isnull().any():
            raise ValueError("Target column contains non-numeric values.")
        target = target_series.astype(int)

    return features, target


def export_processed_dataset(
    features: pd.DataFrame,
    target: Optional[pd.Series],
    path: Optional[Path] = None,
) -> Path:
    """Persist the processed features (and optional target) for transparency."""
    export_path = Path(path) if path else PROCESSED_EXPORT_PATH
    export_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = features.copy()
    if target is not None:
        output_df[TARGET_COLUMN] = target

    output_df.to_csv(export_path, index=False)
    return export_path

