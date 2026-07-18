"""
hyperparameter_tuning.py

Hyperparameter optimization (Optuna, TPE sampler) for LightGBM, Random
Forest and Gradient Boosting.

Corresponds to the notebook section: "T2 - Otimização de Hiperparâmetros".
"""

import optuna
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_TRIALS = 30

# Hardcoded best hyperparameters found by a previous Optuna run. These are
# used as a fast-path default so `model_evaluation.py` can be run without
# re-running the (slow) optimization studies.
DEFAULT_BEST_PARAMS = {
    "LightGBM": {
        "n_estimators": 399,
        "learning_rate": 0.02776214248715833,
        "num_leaves": 140,
        "max_depth": 4,
        "subsample": 0.6100310015463966,
        "colsample_bytree": 0.5547647793859046,
    },
    "RandomForest": {
        "n_estimators": 383,
        "max_depth": 13,
        "min_samples_split": 4,
        "min_samples_leaf": 7,
        "max_features": None,
    },
    "GradientBoosting": {
        "n_estimators": 258,
        "learning_rate": 0.028717221451929068,
        "max_depth": 5,
        "min_samples_split": 16,
        "min_samples_leaf": 8,
        "subsample": 0.5542692739224276,
    },
}


def make_objective_lgbm(X_train, y_train, kf, seed: int = SEED):
    """Build the Optuna objective function for LightGBM."""

    def objective_lgbm(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": seed,
            "verbose": -1,
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgb.LGBMRegressor(**params)),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return scores.mean()

    return objective_lgbm


def make_objective_rf(X_train, y_train, kf, seed: int = SEED):
    """Build the Optuna objective function for Random Forest."""

    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": seed,
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(**params)),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return scores.mean()

    return objective_rf


def make_objective_gb(X_train, y_train, kf, seed: int = SEED):
    """Build the Optuna objective function for Gradient Boosting."""

    def objective_gb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": seed,
        }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(**params)),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return scores.mean()

    return objective_gb


def run_optimization(X_train, y_train, n_trials: int = N_TRIALS, seed: int = SEED) -> dict:
    """
    Run Optuna studies for LightGBM, Random Forest and Gradient Boosting.

    Returns
    -------
    best_params_dict : dict
        model_name -> best hyperparameters found.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_params_dict = {}

    print("Iniciando Otimização...")
    sampler_factory = lambda: optuna.samplers.TPESampler(seed=seed)

    print("\nA otimizar LightGBM...")
    study_lgbm = optuna.create_study(direction="maximize", sampler=sampler_factory())
    study_lgbm.optimize(make_objective_lgbm(X_train, y_train, kf, seed), n_trials=n_trials)
    best_params_dict["LightGBM"] = study_lgbm.best_params
    print(f"Melhor MAE (negativo) LightGBM: {study_lgbm.best_value:.4f}")

    print("\nA otimizar Random Forest...")
    study_rf = optuna.create_study(direction="maximize", sampler=sampler_factory())
    study_rf.optimize(make_objective_rf(X_train, y_train, kf, seed), n_trials=n_trials)
    best_params_dict["RandomForest"] = study_rf.best_params
    print(f"Melhor MAE (negativo) Random Forest: {study_rf.best_value:.4f}")

    print("\nA otimizar Gradient Boosting...")
    study_gb = optuna.create_study(direction="maximize", sampler=sampler_factory())
    study_gb.optimize(make_objective_gb(X_train, y_train, kf, seed), n_trials=n_trials)
    best_params_dict["GradientBoosting"] = study_gb.best_params
    print(f"Melhor MAE (negativo) Gradient Boosting: {study_gb.best_value:.4f}")

    print("\n=== RESUMO DOS MELHORES HIPERPARÂMETROS ===")
    for model_name, params in best_params_dict.items():
        print(f"\n{model_name}: {params}")

    return best_params_dict