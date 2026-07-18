"""
data_preprocessing.py

Handles dataset download, cleaning, feature/target split, preprocessing
pipeline construction, and the train/test split.

Corresponds to the notebook sections:
    - "Fazendo download do dataset e carregando para um DataFrame"
    - "2. Pré-processamento dos dados"
"""

import os

import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

KAGGLE_DATASET = "khanghunhnguyntrng/football-players-transfer-fee-prediction-dataset"
CSV_FILENAME = "final_data.csv"

# Columns that leak the target or are pure identifiers and must not be used
# as model features.
COLS_TO_DROP = [
    "player",
    "name",
    "position",
    "highest_value",
    "team",
    "current_value",
    "appearance",
    "days_injured",
]

MIN_MINUTES_PLAYED = 270  # ~3 full matches


def download_data(dataset: str = KAGGLE_DATASET, csv_filename: str = CSV_FILENAME) -> pd.DataFrame:
    """Download the dataset from Kaggle Hub and load it into a DataFrame."""
    path = kagglehub.dataset_download(dataset)
    csv_path = os.path.join(path, csv_filename)
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame, min_minutes_played: int = MIN_MINUTES_PLAYED):
    """
    Remove players with too little playing time and split the data into
    features (X) and a log-transformed target (y).

    Returns
    -------
    df_clean : pd.DataFrame
        The filtered DataFrame (still containing the target/identifier columns).
    X : pd.DataFrame
        Feature matrix used for modeling.
    y : pd.Series
        Log1p-transformed target (current_value).
    """
    df_clean = df[df["minutes played"] >= min_minutes_played].copy()

    y = np.log1p(df_clean["current_value"])

    cols_to_drop = list(COLS_TO_DROP)
    if "general_position" in df_clean.columns:
        cols_to_drop.append("general_position")

    X = df_clean.drop(columns=cols_to_drop)

    return df_clean, X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that standard-scales numerical features and
    one-hot-encodes categorical features.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    return preprocessor


def split_data(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Perform the initial train/test split (done before any CV/tuning)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Convenience wrapper that runs the full preprocessing pipeline:
    clean -> build preprocessor -> split.
    """
    df_clean, X, y = clean_data(df)

    print("Features usadas no modelo:")
    print(X.columns.tolist())
    print("\nQuantidade de features:", X.shape[1])

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return {
        "df_clean": df_clean,
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }