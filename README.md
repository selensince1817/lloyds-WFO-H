# Workforce Preference Predictor

Machine-learning prototype originally built for the 2022 Lloyds Banking Group Insight Internship.  
The refreshed repo provides a lightweight, reproducible pipeline that predicts whether an employee
is likely to prefer working from home (WFH) or working from the office (WFO) based on qualitative
signals (motivation, wellbeing, commute, etc.).

## Project Highlights
- Random Forest classifier trained on 200+ labelled questionnaire responses.
- Reusable data/preprocessing module with a scikit-learn `Pipeline`.
- CLI tooling for both training (`scripts/train.py`) and batch inference (`scripts/predict.py`).
- Clean repo layout so recruiters can browse notebooks, code, and outputs independently.

## Repository Structure

```
.
├── data/
│   ├── raw/                # Source CSVs (kept under version control)
│   ├── processed/          # Generated features (gitignored, recreate via train script)
│   └── examples/           # Sample inputs for prediction
├── models/                 # Saved pipelines (gitignored, recreated locally)
├── notebooks/              # Exploratory notebooks (EDA, storytelling)
├── reports/                # Visual artefacts & metrics (tree plot, latest scores, etc.)
├── scripts/                # Typer-based CLIs for training + inference
├── src/wfh_pref/           # Python package with data + pipeline logic
├── requirements.txt
└── README.md
```

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Heads-up:** Graphviz is required to regenerate tree diagrams.  
> macOS: `brew install graphviz` · Ubuntu: `sudo apt install graphviz`.

## Training a Fresh Model

```bash
python3 scripts/train.py \
  --test-size 0.25 \
  --n-estimators 400 \
  --random-state 42
```

This command will:
1. Load `data/raw/WFH_WFO_dataset.csv`.
2. Clean/standardise feature columns (drop IDs, fix typos, normalise text).
3. Build a preprocessing + Random Forest pipeline.
4. Write the trained pipeline to `models/random_forest.pkl`.
5. Export metrics to `reports/latest_metrics.json` and the processed dataset to `data/processed/training_features.csv`.

## Running Predictions

Use the provided template in `data/examples/sample_applicants.csv` (same schema as the raw dataset minus the target column). Populate one row per employee, then run:

```bash
python3 scripts/predict.py data/examples/sample_applicants.csv \
  --model-path models/random_forest.pkl \
  --output reports/predictions.csv
```

The CLI prints each employee’s predicted class (`WFH`/`WFO`) alongside the WFH probability, and optionally writes results to CSV.

## Notebook

`notebooks/01_eda.ipynb` preserves the original exploratory analysis (feature exploration, grid-search experiments, early plots).  
The notebook now imports helper functions from `src/wfh_pref` so it stays in sync with the production pipeline.

## Roadmap / Nice-to-haves
- Hyperparameter grid search or nested CV baked into `scripts/train.py`.
- Model cards inside `reports/` for quick recruiter-friendly summaries.
- FastAPI endpoint to expose the predictor as a minimal REST service.
- Unit tests for the preprocessing layer (e.g., `pytest src/tests`).

---

This repo now mirrors the narrative from the original internship: a data-first, winning prototype that can be inspected, rerun, or demoed quickly. Feedback welcome! 
