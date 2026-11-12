import argparse
import os
import pickle
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_jsonl
from src.features import (
    extract_features,
    prune_features,
    fit_historical_stats,
    PRIORITY_DOMAIN_SIGNALS,
)

def compute_complete_statistics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:

    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier_score_loss": float(brier_score_loss(y_true, probs)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "precision_score": float(precision_score(y_true, preds, zero_division=0)),
        "recall_score": float(recall_score(y_true, preds, zero_division=0)),
        "calibrated_threshold": float(threshold),
    }

def format_stats_table(metrics: Dict[str, float], title: str = "PIPELINE STATISTICAL DATA") -> str:

    lines = [
        "=" * 65,
        f"{title.center(65)}",
        "=" * 65,
        f"  {'Metric':<25} | {'Value':<15}",
        "-" * 65,
        f"  {'Accuracy':<25} | {metrics.get('accuracy', 0.0)*100:>8.2f}%",
        f"  {'ROC-AUC':<25} | {metrics.get('roc_auc', 0.0)*100:>8.2f}%",
        f"  {'Brier Score Loss':<25} | {metrics.get('brier_score_loss', 0.0):>9.4f}",
        f"  {'F1-Score':<25} | {metrics.get('f1_score', 0.0)*100:>8.2f}%",
        f"  {'Precision':<25} | {metrics.get('precision_score', 0.0)*100:>8.2f}%",
        f"  {'Recall':<25} | {metrics.get('recall_score', 0.0)*100:>8.2f}%",
    ]
    if "calibrated_threshold" in metrics:
        lines.append(f"  {'Calibrated Threshold':<25} | {metrics['calibrated_threshold']:>9.4f}")
    lines.append("=" * 65)
    return "\n".join(lines)

class PLSSoftModel(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components: int = 14, scale: float = 4.0):
        self.n_components = n_components
        self.scale = scale
        self.pls_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSSoftModel":
        n_comp = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n_comp)
        self.pls_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_scores = self.pls_.predict(X).ravel()
        p1 = 1.0 / (1.0 + np.exp(-self.scale * (raw_scores - 0.5)))
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class PokemonBattlePredictionPipeline:

    def __init__(
        self,
        min_correlation: float = 0.018,
        n_splits_cv: int = 10,
        random_state: int = 42,
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
        return {
            "hist_gbdt_1": HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.035,
                max_leaf_nodes=30,
                min_samples_leaf=20,
                l2_regularization=2.5,
                random_state=self.random_state,
            ),
            "hist_gbdt_2": HistGradientBoostingClassifier(
                max_iter=450,
                learning_rate=0.020,
                max_leaf_nodes=40,
                min_samples_leaf=25,
                l2_regularization=3.5,
                random_state=self.random_state + 100,
            ),
            "logistic_regression": LogisticRegression(
                C=0.04,
                max_iter=600,
                solver="lbfgs",
                random_state=self.random_state,
            ),
            "pls_soft_model": PLSSoftModel(
                n_components=14,
                scale=4.0,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=180,
                max_depth=14,
                min_samples_split=8,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                reg_alpha=0.5,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

    def fit(self, train_records: List[Dict[str, Any]], verbose: bool = True) -> "PokemonBattlePredictionPipeline":

        if verbose:
            print(f"Fitting pipeline on {len(train_records)} battle records...")

        self.historical_stats_ = fit_historical_stats(train_records)

        t0 = time.time()
        df_raw = extract_features(
            train_records,
            n_splits=self.n_splits_cv,
            random_state=self.random_state,
        )
        if verbose:
            print(f"  Feature extraction complete in {time.time() - t0:.2f}s ({df_raw.shape[1] - 2} features).")

        df_pruned, self.retained_features_ = prune_features(
            df_raw,
            target_col="player_won",
            min_correlation=self.min_correlation,
            drop_constants=True,
            force_keep_cols=PRIORITY_DOMAIN_SIGNALS,
        )
        if verbose:
            print(f"  Retained {len(self.retained_features_)} high-signal features (pruned {df_raw.shape[1] - 2 - len(self.retained_features_)} noisy/constant features).")

        X = df_pruned[self.retained_features_].values.astype(float)
        y = df_pruned["player_won"].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        skf = StratifiedKFold(n_splits=self.n_splits_cv, shuffle=True, random_state=self.random_state)
        model_names = ["hist_gbdt_1", "hist_gbdt_2", "logistic_regression", "pls_soft_model", "extra_trees", "xgboost"]
        oof_probs = {m: np.zeros(len(y), dtype=float) for m in model_names}
        fold_scaler = StandardScaler()

        if verbose:
            print(f"  Executing {self.n_splits_cv}-Fold CV to calibrate ensemble weights and threshold...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            X_tr_s = fold_scaler.fit_transform(X_tr)
            X_val_s = fold_scaler.transform(X_val)

            fold_models = self._init_models()
            for name, m in fold_models.items():
                if name in {"logistic_regression", "pls_soft_model"}:
                    m.fit(X_tr_s, y_tr)
                    oof_probs[name][val_idx] = m.predict_proba(X_val_s)[:, 1]
                else:
                    m.fit(X_tr, y_tr)
                    oof_probs[name][val_idx] = m.predict_proba(X_val)[:, 1]

        oof_matrix = np.column_stack([oof_probs[m] for m in model_names])
        self.simplex_weights_ = self._optimize_weights(oof_matrix, y)

        if verbose:
            print("  Optimal SLSQP Simplex Weights:")
            for m, w in zip(model_names, self.simplex_weights_):
                print(f"    - {m:<22}: {w:.4f} ({w*100:.1f}%)")

        self.calibrated_power_r_, self.calibrated_threshold_, best_acc = self._calibrate_power_mean_and_threshold(
            oof_matrix, self.simplex_weights_, y
        )
        blend_probs = self._power_mean_blend(oof_matrix, self.simplex_weights_, r=self.calibrated_power_r_)
        self.cv_metrics_ = compute_complete_statistics(y, blend_probs, threshold=self.calibrated_threshold_)
        if verbose:
            print(f"  Calibrated Power Exponent: r = {self.calibrated_power_r_:.2f}")
            print(f"  Calibrated Decision Threshold: t = {self.calibrated_threshold_:.3f}")
            print("\n" + format_stats_table(self.cv_metrics_, title=f"OUT-OF-FOLD CROSS-VALIDATION STATISTICAL DATA ({self.n_splits_cv}-FOLD)") + "\n")

        if verbose:
            print("  Fitting final base models on all training samples...")
        self.scaler_.fit(X)
        X_scaled = self.scaler_.transform(X)

        self.models_ = self._init_models()
        for name, m in self.models_.items():
            if name in {"logistic_regression", "pls_soft_model"}:
                m.fit(X_scaled, y)
            else:
                m.fit(X, y)

        if verbose:
            print("Pipeline successfully fitted and ready for deployment.")

        return self

    def predict_proba(self, test_records: List[Dict[str, Any]]) -> np.ndarray:

        if not self.models_ or not self.retained_features_:
            raise RuntimeError("Pipeline must be fitted before calling predict_proba.")

        df_test_raw = extract_features(test_records, historical_stats=self.historical_stats_)
        df_test_pruned, _ = prune_features(df_test_raw, selected_cols=self.retained_features_)

        X_test = df_test_pruned[self.retained_features_].values.astype(float)
        X_test = np.nan_to_num(X_test, nan=0.0)
        X_test_scaled = self.scaler_.transform(X_test)

        model_probs = []
        for name, m in self.models_.items():
            if name in {"logistic_regression", "pls_soft_model"}:
                p = m.predict_proba(X_test_scaled)[:, 1]
            else:
                p = m.predict_proba(X_test)[:, 1]
            model_probs.append(p)

        prob_matrix = np.column_stack(model_probs)
        w = self.simplex_weights_ if self.simplex_weights_ is not None else np.ones(prob_matrix.shape[1]) / prob_matrix.shape[1]
        r = getattr(self, "calibrated_power_r_", getattr(self, "power_r", 1.0))
        p1 = self._power_mean_blend(prob_matrix, w, r=r)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, test_records: List[Dict[str, Any]]) -> np.ndarray:

        p1 = self.predict_proba(test_records)[:, 1]
        return (p1 >= self.calibrated_threshold_).astype(int)

    @staticmethod
    def _optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n_models = oof_matrix.shape[1]
        init_w = np.ones(n_models) / n_models

        def loss_fn(weights):
            w = weights / np.sum(weights)
            p = np.clip(oof_matrix @ w, 1e-12, 1.0 - 1e-12)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        bounds = [(0.0, 1.0)] * n_models
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        return opt_w

    @staticmethod
    def _calibrate_threshold(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        best_t = 0.5
        best_acc = accuracy_score(y_true, (probs >= 0.5).astype(int))

        for t in np.linspace(0.45, 0.55, 101):
            acc = accuracy_score(y_true, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)

        return best_t, best_acc

    @classmethod
    def _power_mean_blend(cls, prob_matrix: np.ndarray, weights: np.ndarray, r: float = 1.0) -> np.ndarray:

        p_safe = np.clip(prob_matrix, 1e-6, 1.0 - 1e-6)
        if abs(r - 1.0) < 1e-5:
            return np.clip(p_safe @ weights, 1e-6, 1.0 - 1e-6)
        return np.clip((p_safe ** r @ weights) ** (1.0 / r), 1e-6, 1.0 - 1e-6)

    @classmethod
    def _calibrate_power_mean_and_threshold(
        cls, oof_matrix: np.ndarray, weights: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float, float]:

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

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "PokemonBattlePredictionPipeline":

        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {filepath}")
        return pipeline

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pokémon Battle Prediction Pipeline")
    parser.add_argument("--train-file", type=str, default="/Users/user/Downloads/fds-pokemon-battles-prediction-2026/train.jsonl")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test.jsonl")
    parser.add_argument("--save-pipeline", type=str, default=None, help="Path to save serialized pipeline (.pkl)")
    parser.add_argument("--load-pipeline", type=str, default=None, help="Path to load serialized pipeline (.pkl)")
    parser.add_argument("--predictions-out", type=str, default="predictions.csv", help="Output path for test predictions")
    parser.add_argument("--min-corr", type=float, default=0.018)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    if args.load_pipeline:
        pipeline = PokemonBattlePredictionPipeline.load(args.load_pipeline)
    else:
        train_records = load_jsonl(args.train_file)
        pipeline = PokemonBattlePredictionPipeline(
            min_correlation=args.min_corr,
            n_splits_cv=args.n_splits,
            random_state=42,
        )
        pipeline.fit(train_records, verbose=True)

        if args.save_pipeline:
            pipeline.save(args.save_pipeline)

    if args.test_file:
        print(f"Predicting on {args.test_file}...")
        test_records = load_jsonl(args.test_file)
        probs = pipeline.predict_proba(test_records)[:, 1]
        preds = pipeline.predict(test_records)

        out_df = pd.DataFrame({
            "battle_id": [b.get("battle_id", f"test_{i}") for i, b in enumerate(test_records)],
            "player_won_probability": probs,
            "player_won": preds,
        })
        out_df.to_csv(args.predictions_out, index=False)
        print(f"Saved {len(out_df)} predictions to {args.predictions_out}")

if __name__ == "__main__":
    main()
