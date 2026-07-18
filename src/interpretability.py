"""
interpretability.py

Model interpretability via SHAP (TreeExplainer + summary plots).

Corresponds to the notebook section: "Interpretabilidade dos modelos".
"""

import pandas as pd
import shap


def get_shap_for_model(pipeline, X_test: pd.DataFrame, model_name: str) -> None:
    """
    Compute and plot SHAP values for a fitted (scaler + tree model) pipeline.

    Assumes `pipeline` has named steps 'scaler' and 'model', and that
    'model' is tree-based (compatible with shap.TreeExplainer).
    """
    print(f"\n===== {model_name} ====")

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # --- 1. Explainer ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled_df)

    # --- 2. Adaptar formato SHAP (para regressão) ---
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            shap_output_for_plot = shap_values[0]
        else:
            raise ValueError(
                "Múltiplas saídas SHAP (lista) detectadas. Para regressão univariada, "
                "espera-se um único array 2D. Por favor, ajuste a função ou o modelo."
            )
    elif len(shap_values.shape) == 3:
        raise ValueError(
            "SHAP de classificação multiclasse detectado. O problema atual é de regressão. "
            "Ajuste o modelo ou a função SHAP."
        )
    else:
        shap_output_for_plot = shap_values

    # --- 3. Plot com TODAS as features ---
    shap.summary_plot(
        shap_output_for_plot,
        X_test_scaled_df,
        feature_names=X_test_scaled_df.columns,
        show=True,
    )


def run_interpretability(models: dict, X_test: pd.DataFrame) -> None:
    """
    Run SHAP interpretability for each fitted model in `models`.

    `models` should map model name -> fitted pipeline (with 'scaler' and
    'model' steps), e.g. the output of `model_evaluation.build_final_models`.
    """
    for name, pipeline in models.items():
        get_shap_for_model(pipeline, X_test, name)