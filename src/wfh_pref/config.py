"""Centralised project configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WFH_WFO_dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXAMPLES_DIR = DATA_DIR / "examples"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Target"
DROP_COLUMNS = ["ID", "Name"]

COLUMN_RENAMES = {
    "Same_ofiice_home_location": "Same_office_home_location",
}

CATEGORICAL_COLUMNS = [
    "Occupation",
    "Gender",
    "Same_office_home_location",
    "kids",
    "RM_save_money",
    "RM_quality_time",
    "RM_better_sleep",
    "calmer_stressed",
    "digital_connect_sufficient",
    "RM_job_opportunities",
]

NUMERIC_COLUMNS = [
    "Age",
    "RM_professional_growth",
    "RM_lazy",
    "RM_productive",
    "RM_better_work_life_balance",
    "RM_improved_skillset",
]

ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

DEFAULT_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
METRICS_PATH = REPORTS_DIR / "latest_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importances.csv"
PROCESSED_EXPORT_PATH = PROCESSED_DATA_DIR / "training_features.csv"

