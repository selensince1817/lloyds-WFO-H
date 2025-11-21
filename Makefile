PYTHON ?= python3

.PHONY: install train predict lint

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) scripts/train.py

predict:
	$(PYTHON) scripts/predict.py data/examples/sample_applicants.csv --model-path models/random_forest.pkl

lint:
	ruff check src scripts

