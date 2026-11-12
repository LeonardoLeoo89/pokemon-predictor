"""Feature assembly and public pipeline interfaces."""

import unittest
import numpy as np


class FeatureRefactorTests(unittest.TestCase):
    def test_team_stat_empty_and_known_values(self):
        from src.features.team import _team_stat_features

        empty = _team_stat_features([])
        self.assertTrue(all(value == 0 for value in empty.values()))
        team = [
            {
                "base_hp": 100,
                "base_atk": 80,
                "base_def": 90,
                "base_spd": 70,
                "base_spe": 60,
                "item": "Leftovers",
            },
            {
                "base_hp": 80,
                "base_atk": 100,
                "base_def": 90,
                "base_spd": 50,
                "base_spe": 40,
                "item": None,
            },
        ]
        result = _team_stat_features(team)
        self.assertEqual(result["p1_mean_hp"], 90)
        self.assertEqual(result["p1_std_attack"], 10)
        self.assertEqual(result["p1_mean_spec_bulk"], 55)
        self.assertEqual(result["p1_leftovers_count"], 1)

    def test_historical_oof_excludes_each_encoded_row_and_keeps_order(self):
        from unittest.mock import patch
        from src.features import extraction as features

        rows = [{"battle_id": i, "player_won": i % 2} for i in range(12)]

        def stats(training):
            return {row["battle_id"] for row in training}

        def extract(row, historical_stats):
            self.assertNotIn(row["battle_id"], historical_stats)
            return {"battle_id": row["battle_id"]}

        with (
            patch.object(features, "fit_historical_stats", side_effect=stats),
            patch.object(
                features, "extract_single_battle_features", side_effect=extract
            ),
        ):
            actual = features._historical_oof_rows(rows, 3, 42)
        self.assertEqual([row["battle_id"] for row in actual], list(range(12)))


class PublicInterfaceTests(unittest.TestCase):
    def test_metric_import_remains_compatible(self):
        from src.pipeline import compute_complete_statistics
        from src.evaluation import compute_complete_statistics as implementation

        self.assertIs(compute_complete_statistics, implementation)
        result = implementation(np.array([0, 1]), np.array([0.2, 0.8]))
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["brier_score_loss"], 0.04)

    def test_cli_defaults(self):
        from src.pipeline import _parse_arguments

        args = _parse_arguments([])
        self.assertEqual(args.n_splits, 10)


if __name__ == "__main__":
    unittest.main()
