"""
model_evaluation.py

Trains the final tuned models, evaluates them on the holdout test set
(both in log-space and back-transformed to millions of euros), and plots
comparison / scatter / residual charts.

Corresponds to the notebook sections:
    - "4. Treinamento do Modelo Final e Avaliação no Holdout"
    - "Gráfico com barras"
    - "Gráficos de Dispersão (Previsto vs. Real)"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42


def build_final_models(params_lightgbm: dict, params_rf: dict, params_gb: dict,
                        random_state: int = RANDOM_STATE) -> dict:
    """Build (unfitted) final pipelines for each tuned model."""
    best_lightgbm = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMRegressor(**params_lightgbm, random_state=random_state, verbose=-1)),
    ])

    best_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(**params_rf, random_state=random_state)),
    ])

    best_gb = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(**params_gb, random_state=random_state)),
    ])

    return {"LightGBM": best_lightgbm, "RandomForest": best_rf, "GradientBoosting": best_gb}


def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit each model and evaluate it on the holdout set (log-space target).

    Returns a dict with fitted models, predictions and log-space metrics.
    """
    predictions = {}
    log_metrics = {}

    print("\n==== RESULTADOS NO CONJUNTO DE TESTE (HOLDOUT) ====")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        log_metrics[name] = {"MAE": mae, "R2": r2, "RMSE": rmse}

        print(f"\n==== {name.upper()} ====")
        print(f"MAE: {mae:.4f}")
        print(f"R²:  {r2:.4f}")
        print(f"RMSE:{rmse:.4f}")

    return {"models": models, "predictions": predictions, "log_metrics": log_metrics}


def evaluate_real_scale(y_test, predictions: dict) -> dict:
    """
    Back-transform predictions/target from log1p space to millions of euros
    and compute MAE / RMSE / R2 on that scale.

    Returns
    -------
    y_test_real : np.ndarray
        Target values in millions of euros.
    previsoes : dict
        model_name -> predictions in millions of euros.
    metricas : dict
        model_name -> {MAE, RMSE, R2} on the millions-of-euros scale.
    """
    sns.set_theme(style="whitegrid")
    y_test_real = np.expm1(y_test) / 1e6

    previsoes = {name: np.expm1(y_pred) / 1e6 for name, y_pred in predictions.items()}

    metricas = {}
    print("==== RESULTADOS NO CONJUNTO DE TESTE (HOLDOUT - Em Milhões) ====")
    for nome, y_pred_real in previsoes.items():
        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)

        metricas[nome] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        print(f"\n==== {nome.upper()} ====")
        print(f"MAE:  {mae:.4f} Milhões")
        print(f"RMSE: {rmse:.4f} Milhões")
        print(f"R²:   {r2:.4f}")

    return y_test_real, previsoes, metricas


def plot_metrics_bar(metricas: dict) -> None:
    """Bar chart comparing MAE/RMSE (left axis) and R2 (right axis) per model."""
    modelos = list(metricas.keys())
    mae_vals = [metricas[m]["MAE"] for m in modelos]
    rmse_vals = [metricas[m]["RMSE"] for m in modelos]
    r2_vals = [metricas[m]["R2"] for m in modelos]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(modelos))
    width = 0.35

    ax1.bar(x - width / 2, mae_vals, width, label="MAE (Milhões)", color="skyblue")
    ax1.bar(x + width / 2, rmse_vals, width, label="RMSE (Milhões)", color="steelblue")

    ax1.set_ylabel("Erro Absoluto / Quadrático (Milhões)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modelos, fontweight="bold")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, r2_vals, color="darkorange", marker="o", linewidth=2, markersize=8, label="R²")
    ax2.set_ylabel("Score R²", fontweight="bold", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1.1)

    lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title("Comparação de Desempenho no Conjunto de Teste", pad=30, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_scatter_pred_vs_real(y_test_real, previsoes: dict) -> None:
    """Scatterplots of predicted vs real value (millions) for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    fig.suptitle("Valor Real vs Valor Previsto (em Milhões)", fontsize=16, fontweight="bold")

    max_val = max(y_test_real.max(), max(preds.max() for preds in previsoes.values()))

    for ax, (nome, y_pred_real) in zip(axes, previsoes.items()):
        sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.6, ax=ax, color="blue", edgecolor="k")
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Previsão Perfeita")

        ax.set_title(nome)
        ax.set_xlabel("Valor Real (Milhões)")
        if ax == axes[0]:
            ax.set_ylabel("Valor Previsto (Milhões)")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_residuals(y_test_real, previsoes: dict) -> None:
    """Residual plots (prediction - real) for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    fig.suptitle("Análise de Resíduos (Previsão - Real)", fontsize=16, fontweight="bold")

    for ax, (nome, y_pred_real) in zip(axes, previsoes.items()):
        residuos = y_pred_real - y_test_real

        sns.scatterplot(x=y_test_real, y=residuos, alpha=0.5, ax=ax, color="purple", edgecolor="k")
        ax.axhline(0, color="r", linestyle="--", lw=2, label="Erro Zero")

        ax.set_title(nome)
        ax.set_xlabel("Valor Real (Milhões)")
        if ax == axes[0]:
            ax.set_ylabel("Resíduos (Milhões)")
        ax.legend()

    plt.tight_layout()
    plt.show()


def run_model_evaluation(best_params_dict: dict, X_train, y_train, X_test, y_test,
                          random_state: int = RANDOM_STATE) -> dict:
    """Convenience wrapper running the full model evaluation sequence."""
    models = build_final_models(
        best_params_dict["LightGBM"],
        best_params_dict["RandomForest"],
        best_params_dict["GradientBoosting"],
        random_state=random_state,
    )

    train_result = train_and_evaluate(models, X_train, y_train, X_test, y_test)
    y_test_real, previsoes, metricas = evaluate_real_scale(y_test, train_result["predictions"])

    plot_metrics_bar(metricas)
    plot_scatter_pred_vs_real(y_test_real, previsoes)
    plot_residuals(y_test_real, previsoes)

    return {
        "models": train_result["models"],
        "predictions": train_result["predictions"],
        "log_metrics": train_result["log_metrics"],
        "y_test_real": y_test_real,
        "previsoes": previsoes,
        "metricas": metricas,
    }