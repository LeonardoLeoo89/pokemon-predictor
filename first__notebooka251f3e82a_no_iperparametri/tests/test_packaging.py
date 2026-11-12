"""Package boundaries and challenge-data provenance."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.features import TYPE_CHART


class PackageTests(unittest.TestCase):
    def test_type_chart_matches_challenge_csv(self):
        with Path("data/type_chart_gen2.csv").open() as stream:
            provided = {
                (row["attacking_type"].lower(), row["defending_type"].lower()): float(
                    row["multiplier"]
                )
                for row in csv.DictReader(stream)
            }
        embedded = {
            (a, d): float(value)
            for a, targets in TYPE_CHART.items()
            for d, value in targets.items()
        }
        for pair in provided.keys() | embedded.keys():
            self.assertEqual(provided.get(pair, 1.0), embedded.get(pair, 1.0), pair)

    def test_package_exports_and_shared_species_cache(self):
        import src
        from src import features
        from src.features import opponent

        self.assertTrue(all(hasattr(src, name) for name in src.__all__))
        self.assertIs(features.INTERNAL_SPECIES_CACHE, opponent.INTERNAL_SPECIES_CACHE)

    def test_default_training_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(
                "src",
                root / "src",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            (root / "tests").mkdir()
            shutil.copy("tests/regression_guard.py", root / "tests/regression_guard.py")
            (root / "environment").mkdir()
            shutil.copy("environment/uv.lock", root / "environment/uv.lock")
            (root / "local_development").mkdir()
            (root / "local_development/regression-baseline").symlink_to(
                Path("local_development/regression-baseline").resolve(),
                target_is_directory=True,
            )
            (root / "data").symlink_to(Path("data").resolve(), target_is_directory=True)
            env = {**os.environ, "PYTHONPATH": str(root)}
            check = "import src, sys; assert 'src.pipeline' not in sys.modules; from src import PokemonBattlePredictionPipeline; assert 'matplotlib' not in sys.modules"
            subprocess.run(
                [sys.executable, "-c", check],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [sys.executable, "tests/regression_guard.py", "verify"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Regression verify passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
