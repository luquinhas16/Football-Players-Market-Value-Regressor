"""
main.py

Orchestrates the full pipeline:
    1. Data download & EDA
    2. Preprocessing (cleaning, feature/target split, train/test split)
    3. Spot-checking of baseline models
    4. Hyperparameter tuning (Optuna) for LightGBM / RandomForest / GradientBoosting
    5. Final model training & evaluation on the holdout set
    6. Model interpretability (SHAP)

Usage
-----
    python main.py                 # uses cached/default hyperparameters (fast)
    python main.py --tune          # re-runs Optuna hyperparameter search (slow)
    python main.py --skip-eda      # skip the EDA plots/prints
"""

import argparse
import warnings

import data_preprocessing as dp
import eda
import spot_checking as sc
import hyperparameter_tuning as ht
import model_evaluation as me
import interpretability as it

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Football player transfer-fee prediction pipeline")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter optimization instead of using the cached best params.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=ht.N_TRIALS,
        help="Number of Optuna trials per model (only used with --tune).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the exploratory data analysis step.",
    )
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Skip the SHAP interpretability step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Download data
    print("=" * 60)
    print("1. DOWNLOAD & LOAD DATA")
    print("=" * 60)
    df = dp.download_data()

    # 2. EDA
    if not args.skip_eda:
        print("=" * 60)
        print("2. EDA")
        print("=" * 60)
        df = eda.run_eda(df)

    # 3. Preprocessing
    print("=" * 60)
    print("3. PREPROCESSING")
    print("=" * 60)
    prep = dp.run_preprocessing(df, random_state=RANDOM_STATE)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    preprocessor = prep["preprocessor"]

    # 4. Spot-checking
    print("=" * 60)
    print("4. SPOT-CHECKING")
    print("=" * 60)
    sc.run_spot_checking(X_train, y_train, preprocessor, random_state=RANDOM_STATE)

    # 5. Hyperparameter tuning
    print("=" * 60)
    print("5. HYPERPARAMETER TUNING")
    print("=" * 60)
    if args.tune:
        # Optuna tuning works on the raw numeric/categorical-encoded features,
        # matching the notebook's approach (StandardScaler only, no explicit
        # OneHotEncoder step) — this assumes X_train's categorical columns
        # have already been handled upstream if you use tree models directly.
        best_params_dict = ht.run_optimization(X_train, y_train, n_trials=args.n_trials, seed=RANDOM_STATE)
    else:
        print("Usando hiperparâmetros previamente otimizados (use --tune para reotimizar).")
        best_params_dict = ht.DEFAULT_BEST_PARAMS

    # 6. Final model training & evaluation
    print("=" * 60)
    print("6. MODEL EVALUATION")
    print("=" * 60)
    eval_result = me.run_model_evaluation(
        best_params_dict, X_train, y_train, X_test, y_test, random_state=RANDOM_STATE
    )

    # 7. Interpretability
    if not args.skip_interpretability:
        print("=" * 60)
        print("7. INTERPRETABILITY")
        print("=" * 60)
        it.run_interpretability(eval_result["models"], X_test)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()