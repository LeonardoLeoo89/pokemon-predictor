"""Train, calibrate and apply the three-model Pokémon battle ensemble."""

import argparse
import os
import pickle
import random
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_jsonl
from src.reproducibility import DEFAULT_SEED
from src.evaluation import compute_complete_statistics, format_stats_table
from src.features import (
    extract_features,
    fit_historical_stats,
    STATIC_FEATURES,
    register_species_from_battle,
)


class PokemonBattlePredictionPipeline:
    """Coordinate historical features, base learners and probability blending."""

    def __init__(
        self,
        min_correlation: float = 0.018,
        n_splits_cv: int = 10,
        random_state: int = DEFAULT_SEED,
        power_r: float = 1.0,
    ):
        self.min_correlation = min_correlation
        self.n_splits_cv = n_splits_cv
        self.random_state = random_state
        self.power_r = power_r

        self.historical_stats_: Optional[Dict[str, Any]] = None
        self.retained_features_: List[str] = []
        self.scaler_: StandardScaler = StandardScaler()
        self.models_: Dict[str, Any] = {}
        self.simplex_weights_: Optional[np.ndarray] = None
        self.calibrated_power_r_: float = power_r
        self.calibrated_threshold_: float = 0.5
        self.cv_results_: Optional[pd.DataFrame] = None
        self.cv_metrics_: Optional[Dict[str, float]] = None

    def _init_models(self) -> Dict[str, Any]:
        """Create the three base learners."""
        return {
            "hist_gbdt_1": HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.005420531605905941,
                max_depth=5,
                max_leaf_nodes=8,
                min_samples_leaf=51,
                max_features=0.6082153918960502,
                max_bins=255,
                l2_regularization=0.24396976647454727,
                random_state=self.random_state,
            ),
            "hist_gbdt_2": HistGradientBoostingClassifier(
                max_iter=450,
                learning_rate=0.013001505004011543,
                max_depth=5,
                max_leaf_nodes=25,
                min_samples_leaf=41,
                max_features=0.7022232070412249,
                max_bins=63,
                l2_regularization=0.0,
                random_state=self.random_state + 100,
            ),
            "logistic_regression": LogisticRegression(
                C=0.019814620165945537,
                max_iter=10000,
                tol=1e-5,
                solver="lbfgs",
                random_state=self.random_state,
            ),
        }

    def fit(
        self, train_records: List[Dict[str, Any]], verbose: bool = True
    ) -> "PokemonBattlePredictionPipeline":
        """Prepare features, collect OOF predictions, calibrate and refit base models."""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        if verbose:
            print(f"Fitting pipeline on {len(train_records)} battle records...")
        X, y = self._prepare_training_data(train_records, verbose)
        model_names, oof_matrix = self._collect_oof_predictions(X, y, verbose)
        self._calibrate_ensemble(oof_matrix, y, model_names, verbose)
        self._fit_final_models(X, y, verbose)
        if verbose:
            print("Training finished.")
        return self

    @staticmethod
    def _model_input(name, raw, scaled):
        """Only logistic regression consumes standardized features."""
        return scaled if name == "logistic_regression" else raw

    def _prepare_training_data(self, train_records, verbose):
        """Fit historical estimates and assemble the training matrix."""
        self.historical_stats_ = fit_historical_stats(train_records)

        t0 = time.time()
        df_raw = extract_features(
            train_records,
            n_splits=self.n_splits_cv,
            random_state=self.random_state,
        )
        if verbose:
            print(
                f"  Feature extraction complete in {time.time() - t0:.2f}s ({df_raw.shape[1] - 2} features)."
            )

        self.retained_features_ = list(STATIC_FEATURES)
        if verbose:
            print(f"  Using {len(self.retained_features_)} retained features.")

        X = df_raw[self.retained_features_].values.astype(float)
        y = df_raw["player_won"].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        return X, y

    def _collect_oof_predictions(self, X, y, verbose):
        """Fit fresh models per fold and restore predictions to row order."""
        skf = StratifiedKFold(
            n_splits=self.n_splits_cv, shuffle=True, random_state=self.random_state
        )
        model_names = list(self._init_models().keys())
        oof_probs = {m: np.zeros(len(y), dtype=float) for m in model_names}
        fold_scaler = StandardScaler()

        if verbose:
            print(
                f"  Executing {self.n_splits_cv}-Fold CV to calibrate ensemble weights and threshold..."
            )

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            X_tr_s = fold_scaler.fit_transform(X_tr)
            X_val_s = fold_scaler.transform(X_val)

            fold_models = self._init_models()
            for name, model in fold_models.items():
                model.fit(self._model_input(name, X_tr, X_tr_s), y_tr)
                validation = self._model_input(name, X_val, X_val_s)
                oof_probs[name][val_idx] = model.predict_proba(validation)[:, 1]

        return model_names, np.column_stack([oof_probs[m] for m in model_names])

    def _calibrate_ensemble(self, oof_matrix, y, model_names, verbose):
        """Select blend weights, power and decision threshold."""
        self.simplex_weights_ = self._optimize_weights(oof_matrix, y)

        if verbose:
            print("  Blend weights:")
            for m, w in zip(model_names, self.simplex_weights_):
                print(f"    - {m:<22}: {w:.4f} ({w * 100:.1f}%)")

        self.calibrated_power_r_, self.calibrated_threshold_, _ = (
            self._calibrate_power_mean_and_threshold(
                oof_matrix, self.simplex_weights_, y
            )
        )
        blend_probs = self._power_mean_blend(
            oof_matrix, self.simplex_weights_, r=self.calibrated_power_r_
        )
        self.cv_metrics_ = compute_complete_statistics(
            y, blend_probs, threshold=self.calibrated_threshold_
        )
        # Populate comparison table matching Table 1 of the report
        name_map = {
            "logistic_regression": "Logistic Regression",
            "hist_gbdt_1": "HistGBDT 1",
            "hist_gbdt_2": "HistGBDT 2",
        }
        table_rows = []
        for i, name in enumerate(model_names):
            m_stats = compute_complete_statistics(y, oof_matrix[:, i], threshold=0.5)
            table_rows.append({
                "Model / Pipeline": name_map.get(name, name),
                "Acc (%)": round(m_stats["accuracy"] * 100, 2),
                "AUC (%)": round(m_stats["roc_auc"] * 100, 2),
                "Brier": round(m_stats["brier_score_loss"], 4),
                "F1 (%)": round(m_stats["f1_score"] * 100, 2),
            })

        avg_probs = oof_matrix.mean(axis=1)
        avg_stats = compute_complete_statistics(y, avg_probs, threshold=0.5)
        table_rows.append({
            "Model / Pipeline": "Average",
            "Acc (%)": round(avg_stats["accuracy"] * 100, 2),
            "AUC (%)": round(avg_stats["roc_auc"] * 100, 2),
            "Brier": round(avg_stats["brier_score_loss"], 4),
            "F1 (%)": round(avg_stats["f1_score"] * 100, 2),
        })

        simplex_linear_probs = oof_matrix @ self.simplex_weights_
        simplex_stats = compute_complete_statistics(y, simplex_linear_probs, threshold=0.5)
        table_rows.append({
            "Model / Pipeline": "Optimized Weighted Average (SLSQP)",
            "Acc (%)": round(simplex_stats["accuracy"] * 100, 2),
            "AUC (%)": round(simplex_stats["roc_auc"] * 100, 2),
            "Brier": round(simplex_stats["brier_score_loss"], 4),
            "F1 (%)": round(simplex_stats["f1_score"] * 100, 2),
        })

        table_rows.append({
            "Model / Pipeline": "Tuned Weighted Average (Hölder)",
            "Acc (%)": round(self.cv_metrics_["accuracy"] * 100, 2),
            "AUC (%)": round(self.cv_metrics_["roc_auc"] * 100, 2),
            "Brier": round(self.cv_metrics_["brier_score_loss"], 4),
            "F1 (%)": round(self.cv_metrics_["f1_score"] * 100, 2),
        })
        self.cv_results_ = pd.DataFrame(table_rows)

        if verbose:
            print(f"  Calibrated Power Exponent: r = {self.calibrated_power_r_:.2f}")
            print(
                f"  Calibrated Decision Threshold: t = {self.calibrated_threshold_:.3f}"
            )
            print(
                "\n"
                + format_stats_table(
                    self.cv_metrics_,
                    title=f"Cross-validation calibration metrics ({self.n_splits_cv}-FOLD)",
                )
                + "\n"
            )
            print(f"10-Fold CV Performance Comparison:\n{self.cv_results_.to_string(index=False)}\n")

    def cv_summary_table(self) -> pd.DataFrame:
        """Return the 10-Fold CV comparison table matching Table 1 of the report."""
        if self.cv_results_ is None:
            raise RuntimeError("Pipeline must be fitted before calling cv_summary_table.")
        return self.cv_results_

    def _fit_final_models(self, X, y, verbose):
        """Fit the production scaler and base learners on all training rows."""
        if verbose:
            print("  Fitting base models on all training rows...")
        self.scaler_.fit(X)
        X_scaled = self.scaler_.transform(X)

        self.models_ = self._init_models()
        for name, model in self.models_.items():
            model.fit(self._model_input(name, X, X_scaled), y)

    def predict_proba(self, test_records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts features for test records and returns continuous ensemble probability predictions.
        """
        if not self.models_ or not self.retained_features_:
            raise RuntimeError("Pipeline must be fitted before calling predict_proba.")

        df_test = extract_features(
            test_records, historical_stats=self.historical_stats_
        )
        X_test = df_test[self.retained_features_].values.astype(float)
        X_test = np.nan_to_num(X_test, nan=0.0)
        X_test_scaled = self.scaler_.transform(X_test)

        model_probs = []
        for name, model in self.models_.items():
            values = self._model_input(name, X_test, X_test_scaled)
            model_probs.append(model.predict_proba(values)[:, 1])

        prob_matrix = np.column_stack(model_probs)
        w = (
            self.simplex_weights_
            if self.simplex_weights_ is not None
            else np.ones(prob_matrix.shape[1]) / prob_matrix.shape[1]
        )
        r = getattr(self, "calibrated_power_r_", getattr(self, "power_r", 1.0))
        p1 = self._power_mean_blend(prob_matrix, w, r=r)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, test_records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Returns binary classification predictions using the calibrated decision threshold.
        """
        p1 = self.predict_proba(test_records)[:, 1]
        return (p1 >= self.calibrated_threshold_).astype(int)

    @staticmethod
    def _optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Minimize log loss on the probability simplex."""
        n_models = oof_matrix.shape[1]
        init_w = np.ones(n_models) / n_models

        def loss_fn(weights):
            w = weights / np.sum(weights)
            p = np.clip(oof_matrix @ w, 1e-12, 1.0 - 1e-12)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints
        )
        opt_w = res.x / np.sum(res.x)
        return opt_w

    @staticmethod
    def _calibrate_threshold(
        probs: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float]:
        """Find the accuracy-maximizing threshold on the candidate grid."""
        best_t = 0.5
        best_acc = accuracy_score(y_true, (probs >= 0.5).astype(int))

        for t in np.linspace(0.45, 0.55, 101):
            acc = accuracy_score(y_true, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)

        return best_t, best_acc

    @classmethod
    def _power_mean_blend(
        cls, prob_matrix: np.ndarray, weights: np.ndarray, r: float = 1.0
    ) -> np.ndarray:
        """Computes generalized power-mean (Hölder mean) ensemble probabilities: p_w(r) = (sum w_i * p_i^r)^(1/r)."""
        p_safe = np.clip(prob_matrix, 1e-6, 1.0 - 1e-6)
        if abs(r - 1.0) < 1e-5:
            return np.clip(p_safe @ weights, 1e-6, 1.0 - 1e-6)
        return np.clip((p_safe**r @ weights) ** (1.0 / r), 1e-6, 1.0 - 1e-6)

    @classmethod
    def _calibrate_power_mean_and_threshold(
        cls, oof_matrix: np.ndarray, weights: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float, float]:
        """Jointly searches for optimal power exponent r and decision threshold t on out-of-fold predictions."""
        best_r = 1.0
        best_t = 0.5
        best_acc = 0.0

        for r in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            p = cls._power_mean_blend(oof_matrix, weights, r=r)
            t, acc = cls._calibrate_threshold(p, y_true)
            if acc > best_acc:
                best_acc = acc
                best_r = float(r)
                best_t = float(t)

        return best_r, best_t, best_acc

    def save(self, filepath: str) -> None:
        """Serializes the pipeline object to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "PokemonBattlePredictionPipeline":
        """Loads a serialized pipeline object from disk."""
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {filepath}")
        return pipeline


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Pokémon battle prediction"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="fds-pokemon-battles-prediction-2026/train.jsonl",
    )
    parser.add_argument(
        "--test-file", type=str, default=None, help="Path to test.jsonl"
    )
    parser.add_argument(
        "--save-pipeline",
        type=str,
        default=None,
        help="Path to save serialized pipeline (.pkl)",
    )
    parser.add_argument(
        "--load-pipeline",
        type=str,
        default=None,
        help="Path to load serialized pipeline (.pkl)",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="predictions.csv",
        help="Output path for test predictions",
    )
    parser.add_argument("--min-corr", type=float, default=0.018)
    parser.add_argument("--n-splits", type=int, default=10)
    return parser.parse_args(argv)


def _load_or_fit_pipeline(args):
    """Load the requested artifact or train and optionally save a pipeline."""
    if args.load_pipeline:
        pipeline = PokemonBattlePredictionPipeline.load(args.load_pipeline)
    else:
        train_records = load_jsonl(args.train_file)
        pipeline = PokemonBattlePredictionPipeline(
            min_correlation=args.min_corr,
            n_splits_cv=args.n_splits,
            random_state=DEFAULT_SEED,
        )
        pipeline.fit(train_records, verbose=True)

        if args.save_pipeline:
            pipeline.save(args.save_pipeline)
    return pipeline


def _save_predictions(pipeline, args):
    """Write battle IDs and predicted labels in input order."""
    print(f"Predicting on {args.test_file}...")
    test_records = load_jsonl(args.test_file)

    # Fast cache warm-up so a single feature extraction pass matches Pass 2 exactly.
    for battle in test_records:
        register_species_from_battle(battle)
    probs = pipeline.predict_proba(test_records)
    preds = (probs[:, 1] >= pipeline.calibrated_threshold_).astype(int)

    out_df = pd.DataFrame(
        {
            "battle_id": [b.get("battle_id", i) for i, b in enumerate(test_records)],
            "player_won": preds,
        }
    )
    out_df.to_csv(args.predictions_out, index=False)
    print(
        f"Wrote {len(out_df)} predictions to {args.predictions_out}"
    )


def main(argv=None):
    """Run training or loading and inference in that order."""
    args = _parse_arguments(argv)
    pipeline = _load_or_fit_pipeline(args)
    if args.test_file:
        _save_predictions(pipeline, args)


if __name__ == "__main__":
    main()
