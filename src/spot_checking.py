"""
spot_checking.py

Baseline model comparison ("spot-checking") across several regression
algorithms using cross-validation.

Corresponds to the notebook section: "3. Spot checking".
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

RANDOM_STATE = 42


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """Return the dictionary of candidate models to spot-check."""
    return {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LightGBM": LGBMRegressor(random_state=random_state, verbose=-1),
    }


def spot_check(X_train, y_train, preprocessor, random_state: int = RANDOM_STATE, n_splits: int = 10):
    """
    Cross-validate every candidate model (wrapped with the shared
    preprocessor) and collect MAE / RMSE / R2 statistics.

    Returns
    -------
    results : dict
        model_name -> {MAE_mean, MAE_std, RMSE_mean, RMSE_std, R2_mean, R2_std}
    results_raw_mae : dict
        model_name -> array of per-fold MAE scores (used for the boxplot).
    """
    models = get_models(random_state)
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    results = {}
    results_raw_mae = {}

    print("\n===== SPOT-CHECKING =====")

    scoring_metrics = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model),
        ])

        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics, return_train_score=False
        )

        mae_scores = -scores["test_MAE"]
        rmse_scores = -scores["test_RMSE"]
        r2_scores = scores["test_R2"]

        results_raw_mae[name] = mae_scores

        results[name] = {
            "MAE_mean": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "RMSE_mean": rmse_scores.mean(),
            "RMSE_std": rmse_scores.std(),
            "R2_mean": r2_scores.mean(),
            "R2_std": r2_scores.std(),
        }

        print(
            f"{name:18} | MAE: {mae_scores.mean():.4f} (±{mae_scores.std():.4f}) "
            f"| R2: {r2_scores.mean():.4f} | RMSE: {rmse_scores.mean():.4f}"
        )

    return results, results_raw_mae


def plot_spot_check_boxplot(results_raw_mae: dict) -> None:
    """Boxplot comparing per-fold MAE across models."""
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [results_raw_mae[m] for m in results_raw_mae.keys()],
        labels=list(results_raw_mae.keys()),
        showmeans=True,
    )
    plt.title("Spot-checking: Comparação de Modelos (Regressão)")
    plt.xlabel("Modelos")
    plt.ylabel("Erro Absoluto Médio (Log Scale) - Menor é Melhor")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def rank_models(results: dict) -> pd.DataFrame:
    """Sort models by mean MAE (ascending) and return the results DataFrame."""
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="MAE_mean", ascending=True)

    print("\n===== RESULTADOS ORDENADOS =====")
    print(results_df)

    return results_df


def select_best_models(results_df: pd.DataFrame, n: int = 3) -> list:
    """Return the names of the top-n models by MAE."""
    best_models_names = results_df.head(n).index.tolist()
    print("Modelos escolhidos para otimização:")
    print(best_models_names)
    return best_models_names


def run_spot_checking(X_train, y_train, preprocessor, random_state: int = RANDOM_STATE, top_n: int = 3):
    """Convenience wrapper running the full spot-checking sequence."""
    results, results_raw_mae = spot_check(X_train, y_train, preprocessor, random_state)
    plot_spot_check_boxplot(results_raw_mae)
    results_df = rank_models(results)
    best_models_names = select_best_models(results_df, n=top_n)

    return {
        "results": results,
        "results_raw_mae": results_raw_mae,
        "results_df": results_df,
        "best_models_names": best_models_names,
    }