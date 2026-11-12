"""Compare refactored feature preparation with an immutable original source file.

Prepare the reference with:
  git show 76840fa:src/features.py > local_development/features.reference.py
This differential check uses the complete training dataset and performs no model fits.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data_loader import load_jsonl
from src.pipeline import PokemonBattlePredictionPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("local_development/features.reference.py"),
    )
    parser.add_argument("--train", default="data/train.jsonl")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_features", args.reference)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    records = load_jsonl(args.train)
    original.fit_historical_stats(records)
    expected = original.extract_features(records, n_splits=10, random_state=42)
    pipeline = PokemonBattlePredictionPipeline(n_splits_cv=10, random_state=42)
    actual, target = pipeline._prepare_training_data(records, verbose=False)
    assert pipeline.retained_features_ == original.STATIC_FEATURES
    np.testing.assert_array_equal(target, expected.player_won.to_numpy(int))
    reference = np.nan_to_num(
        expected[original.STATIC_FEATURES].to_numpy(float), nan=0.0
    )
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12, equal_nan=True)
    print(
        f"Full feature parity passed: {len(records)} rows × {actual.shape[1]} ordered features; no model training."
    )


if __name__ == "__main__":
    main()
