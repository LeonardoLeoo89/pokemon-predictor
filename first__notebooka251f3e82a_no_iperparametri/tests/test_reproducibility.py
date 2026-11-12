"""Repeatability of training initialization and out-of-fold predictions."""

import random
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification

from src.pipeline import PokemonBattlePredictionPipeline
from src.reproducibility import DEFAULT_SEED


class ReproducibilityTests(unittest.TestCase):
    def test_fit_resets_global_generators(self):
        pipeline = PokemonBattlePredictionPipeline()
        observed = []

        def prepare(records, verbose):
            observed.append((random.random(), np.random.random()))
            return None, None

        with (
            patch.object(pipeline, '_prepare_training_data', side_effect=prepare),
            patch.object(pipeline, '_collect_oof_predictions', return_value=([], None)),
            patch.object(pipeline, '_calibrate_ensemble'),
            patch.object(pipeline, '_fit_final_models'),
        ):
            pipeline.fit([], verbose=False)
            random.random()
            np.random.random()
            pipeline.fit([], verbose=False)
        self.assertEqual(observed[0], observed[1])

    def test_oof_predictions_ignore_ambient_random_state(self):
        X, y = make_classification(n_samples=180, n_features=8, random_state=DEFAULT_SEED)
        results = []
        for seed in (7, 999):
            random.seed(seed)
            np.random.seed(seed)
            pipeline = PokemonBattlePredictionPipeline(n_splits_cv=3)
            names, probabilities = pipeline._collect_oof_predictions(X, y, verbose=False)
            results.append(probabilities)
        self.assertEqual(names, ['hist_gbdt_1', 'hist_gbdt_2', 'logistic_regression'])
        np.testing.assert_array_equal(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
