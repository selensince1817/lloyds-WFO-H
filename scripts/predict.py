from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wfh_pref.config import DEFAULT_MODEL_PATH
from wfh_pref.data import prepare_feature_frame

app = typer.Typer(help="Run batch predictions using the trained WFH preference model.")


@app.command()
def main(
    input_csv: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="CSV containing the workforce questionnaire responses (without the Target column).",
    ),
    model_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model-path",
        "-m",
        help="Path to a trained pipeline produced by scripts/train.py.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save the scored dataframe as CSV.",
    ),
) -> None:
    """Predict WFH vs WFO preference for the provided employees."""
    if not model_path.exists():
        raise typer.BadParameter(
            f"Model not found at {model_path}. Train a model via `python scripts/train.py` first."
        )

    typer.echo(f"📂 Loading rows from {input_csv} ...")
    raw_df = pd.read_csv(input_csv)

    try:
        features, _ = prepare_feature_frame(raw_df, expect_target=False)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"📦 Loading model from {model_path} ...")
    pipeline = joblib.load(model_path)

    typer.echo("🤖 Generating predictions...")
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    results = raw_df.copy()
    results["prediction"] = ["WFH" if pred == 1 else "WFO" for pred in predictions]
    results["wfh_probability"] = probabilities

    for name, pred, prob in zip(results.get("Name", results.index), predictions, probabilities):
        label = "WFH" if pred == 1 else "WFO"
        typer.echo(f" - {name}: {label} ({prob:.2%} WFH probability)")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        typer.echo(f"🗂  Full results written to {output}")


if __name__ == "__main__":
    app()

