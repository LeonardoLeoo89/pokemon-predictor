"""Record once, then compare pipeline outputs in the same Docker environment.

This 240-row/3-fold fixture checks regression, not predictive performance.
The production estimators and their hyperparameters are left unchanged.
Run both modes from the repository root with the datasets mounted under data/.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data_loader import load_jsonl, parse_metadata_dataframe
from src.features import extract_features, fit_historical_stats
from src.pipeline import PokemonBattlePredictionPipeline as Pipeline


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(train_path, test_path, output):
    records = load_jsonl(str(train_path), limit=240)
    test = load_jsonl(str(test_path), limit=12)
    assert len(records) == 240 and len(test) == 12
    stats = fit_historical_stats(records)
    frame = extract_features(records, n_splits=3, random_state=42)
    columns = [name for name in frame if name not in ("battle_id", "player_won")]
    arrays = {
        "features": frame[columns].to_numpy(float),
        "target": frame.player_won.to_numpy(),
        "battle_ids": frame.battle_id.to_numpy(),
    }
    try:
        Pipeline().predict_proba(test)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unfitted pipeline must reject inference")
    metadata = {
        "fixture": {"train_rows": 240, "test_rows": 12, "folds": 3, "seed": 42},
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "lock_sha256": sha256("environment/uv.lock"),
        "columns": columns,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "catboost",
                "lightgbm",
                "xgboost",
            )
        },
        "metadata_columns": parse_metadata_dataframe(test).columns.tolist(),
        "prior": float(stats["global_prior"]),
    }
    # Preserve the original fit, serialization and CLI inference behavior.
    for suffix in ("original",):
        pipeline = Pipeline(n_splits_cv=3, random_state=42)
        pipeline.fit(records, verbose=False)
        values = extract_features(test, historical_stats=pipeline.historical_stats_)
        x = np.nan_to_num(values[pipeline.retained_features_].to_numpy(float), nan=0.0)
        scaled = pipeline.scaler_.transform(x)
        arrays[suffix + "_input"] = x
        arrays[suffix + "_scaled"] = scaled
        arrays[suffix + "_probabilities"] = pipeline.predict_proba(test)
        arrays[suffix + "_predictions"] = pipeline.predict(test)
        arrays[suffix + "_weights"] = pipeline.simplex_weights_
        for name, model in pipeline.models_.items():
            input_values = scaled if name in {"logistic_regression", "bagged_lr"} else x
            arrays[suffix + "_" + name] = model.predict_proba(input_values)
        metadata[suffix] = {
            "retained_features": pipeline.retained_features_,
            "threshold": pipeline.calibrated_threshold_,
            "metrics": pipeline.cv_metrics_,
            "models": {k: repr(v.get_params()) for k, v in pipeline.models_.items()},
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            model_file = temporary / "pipeline.joblib"
            test_file = temporary / "test.jsonl"
            csv_file = temporary / "predictions.csv"
            pipeline.save(str(model_file))
            loaded = Pipeline.load(str(model_file))
            np.testing.assert_array_equal(
                loaded.predict_proba(test), arrays[suffix + "_probabilities"]
            )
            test_file.write_text("".join(json.dumps(record) + "\n" for record in test))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.pipeline",
                    "--load-pipeline",
                    str(model_file),
                    "--test-file",
                    str(test_file),
                    "--predictions-out",
                    str(csv_file),
                ],
                stdout=subprocess.DEVNULL,
                check=True,
            )
            (output / (suffix + ".csv")).write_bytes(csv_file.read_bytes())
    return arrays, metadata


def compare_metadata(expected, actual, prefix="metadata"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), prefix
        for key in expected:
            compare_metadata(expected[key], actual[key], prefix + "." + key)
    elif isinstance(expected, float):
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-7, equal_nan=True, err_msg=prefix
        )
    else:
        assert expected == actual, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("local_development/regression-baseline")
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/test.jsonl"))
    args = parser.parse_args()
    if args.mode == "record" and args.baseline.exists():
        parser.error("Baseline exists; it must never be overwritten automatically")
    if args.mode == "verify" and not (args.baseline / "metadata.json").exists():
        parser.error("Record an original baseline before verification")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            arrays, metadata = capture(args.train, args.test, output)
        if args.mode == "record":
            args.baseline.mkdir(parents=True)
            np.savez_compressed(args.baseline / "arrays.npz", **arrays)
            (args.baseline / "metadata.json").write_text(json.dumps(metadata, indent=2))
            for path in output.glob("*.csv"):
                (args.baseline / path.name).write_bytes(path.read_bytes())
        else:
            compare_metadata(
                json.loads((args.baseline / "metadata.json").read_text()), metadata
            )
            with np.load(args.baseline / "arrays.npz", allow_pickle=False) as expected:
                assert set(expected.files) == set(arrays)
                for name, value in arrays.items():
                    reference = expected[name]
                    assert value.shape == reference.shape, name
                    if reference.dtype.kind in "biu":
                        np.testing.assert_array_equal(value, reference, err_msg=name)
                    elif name == "features" or name.endswith(("_input", "_scaled")):
                        np.testing.assert_allclose(
                            value,
                            reference,
                            rtol=0,
                            atol=1e-12,
                            equal_nan=True,
                            err_msg=name,
                        )
                    else:
                        np.testing.assert_allclose(
                            value,
                            reference,
                            rtol=0,
                            atol=1e-7,
                            equal_nan=True,
                            err_msg=name,
                        )
            for path in output.glob("*.csv"):
                assert path.read_bytes() == (args.baseline / path.name).read_bytes(), (
                    path.name
                )
    print(
        f"Regression {args.mode} passed: features, preprocessing, models, metrics and CLI CSV."
    )


if __name__ == "__main__":
    main()
