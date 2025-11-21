from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wfh_pref.config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    NUMERIC_COLUMNS,
    PROCESSED_EXPORT_PATH,
    RAW_DATA_PATH,
)
from wfh_pref.data import export_processed_dataset, load_raw_dataset, prepare_feature_frame
from wfh_pref.pipeline import build_model_pipeline

app = typer.Typer(help="Train the WFH vs WFO preference classifier.")


def _save_metrics(payload: dict, path: Path = METRICS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _save_feature_importances(pipeline, path: Path = FEATURE_IMPORTANCE_PATH) -> Optional[Path]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    feature_names = list(cat_feature_names) + NUMERIC_COLUMNS

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(path, index=False)
    return path


@app.command()
def main(
    raw_data: Path = typer.Option(
        RAW_DATA_PATH,
        "--raw-data",
        help="Path to the original workforce preference dataset.",
    ),
    model_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model-path",
        help="Where to store the trained pipeline.",
    ),
    test_size: float = typer.Option(0.25, help="Test split proportion."),
    random_state: int = typer.Option(42, help="Random seed used across splits."),
    n_estimators: int = typer.Option(400, help="Number of trees in the Random Forest."),
    max_depth: Optional[int] = typer.Option(
        None,
        help="Optional depth constraint for each tree. Leave empty for fully grown trees.",
    ),
    export_processed: bool = typer.Option(
        True,
        help="Export the processed training set to data/processed for transparency.",
    ),
):
    """Train, evaluate, and persist the model pipeline."""
    typer.echo("📦 Loading dataset...")
    raw_df = load_raw_dataset(raw_data)

    typer.echo("🧹 Preparing features...")
    features, target = prepare_feature_frame(raw_df, expect_target=True)

    if export_processed:
        export_path = export_processed_dataset(features, target, PROCESSED_EXPORT_PATH)
        typer.echo(f"↳ Processed dataset saved to {export_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    typer.echo("🌲 Training Random Forest pipeline...")
    pipeline = build_model_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)

    typer.echo("📈 Evaluating...")
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }
    report = classification_report(
        y_test,
        predictions,
        target_names=["WFO", "WFH"],
        output_dict=True,
    )

    for name, value in metrics.items():
        typer.echo(f"{name:>10}: {value:.3f}")

    metrics_payload = {
        "metrics": metrics,
        "classification_report": report,
    }
    metrics_path = _save_metrics(metrics_payload)
    typer.echo(f"📝 Metrics written to {metrics_path}")

    feature_path = _save_feature_importances(pipeline)
    typer.echo(f"⭐ Feature importances saved to {feature_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    typer.echo(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    app()

