# Pokémon Battle Prediction — Kaggle Submissions

This repository contains the source code for the models evaluated in the Kaggle competition:

- **`first__notebooka251f3e82a_no_iperparametri`**:
  Primary reference pipeline (`notebooka251f3e82a - No iperparametri`). Employs the fixed 135-feature schema across five domain pillars and a 3-model ensemble (two HistGradientBoosting classifiers and Logistic Regression) with robust baseline hyperparameters, calibrated via SLSQP simplex blending and Hölder power-mean ensembling.

- **`second__notebooka251f3e82a_iperparametri`**:
  Identical architecture and feature representation as `first`, with base learner hyperparameters optimized via Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE) sampler.

- **`third__PokemonPredictor_v10`**:
  An earlier development iteration combining a 6-model ensemble (HistGBDT 1 & 2, Logistic Regression, PLS, ExtraTrees, XGBoost) with dynamic correlation screening and runtime feature pruning, containing experimental features that were streamlined in later versions.
