# Pokémon Battle Prediction

Machine learning pipeline for Pokémon Generation 2 battle outcome prediction using a three-model ensemble (Histogram Gradient Boosted Trees and Logistic Regression) with domain-specific feature engineering.

## Run

```sh
pip install -r requirements.txt

python -m src.pipeline \
  --train-file data/train.jsonl \
  --test-file data/test.jsonl \
  --predictions-out submission.csv
```

Or using Docker:

```sh
docker build -t pokemon-predictor environment
docker run --rm --cpus 4 --memory 10g -v "$PWD:/app" pokemon-predictor \
  -m src.pipeline --train-file data/train.jsonl --test-file data/test.jsonl --predictions-out submission.csv
```

## Structure

```text
src/
├── features/          # Feature extraction (team stats, type synergy, timeline, opponent tracking)
├── pipeline.py        # End-to-end training, calibration, and inference CLI
├── evaluation.py      # Validation metrics computation
└── reproducibility.py # Seed management
```
