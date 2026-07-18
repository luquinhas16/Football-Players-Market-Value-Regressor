"""
eda.py

Exploratory Data Analysis for the football players transfer-fee dataset.

Corresponds to the notebook section: "1. EDA (Exploratory Data Analysis)".
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def basic_info(df: pd.DataFrame) -> None:
    """Print dtypes/null info plus row/column counts and duplicate count."""
    df.info()
    print("Número de linhas:", df.shape[0])
    print("Número de colunas:", df.shape[1])
    print("Duplicatas:", df.duplicated().sum())


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot the correlation heatmap for all numeric columns."""
    plt.figure(figsize=(15, 8))
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()


def peak_value_percentage(df: pd.DataFrame) -> float:
    """Print/return the % of players currently at their historical peak value."""
    peak_percentage = (df["highest_value"] == df["current_value"]).mean() * 100
    print(f"{peak_percentage:.2f}% de jogadores onde highest_value == current_value")
    return peak_percentage


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot raw and log-scaled distributions of the target (current_value)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(df["current_value"], bins=50, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Raw Distribution of Current Value")
    axes[0].set_xlabel("Current Value (€)")
    axes[0].set_ylabel("Number of Players")

    sns.histplot(df["current_value"], bins=50, kde=True, ax=axes[1], color="green", log_scale=True)
    axes[1].set_title("Log-Scaled Distribution of Current Value")
    axes[1].set_xlabel("Current Value (Log Scale)")
    axes[1].set_ylabel("Number of Players")

    plt.tight_layout()
    plt.show()


def plot_target_boxplot(df: pd.DataFrame) -> None:
    """Boxplot of current_value to highlight elite outliers."""
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df["current_value"], color="orange")
    plt.title("Boxplot of Current Value (Showing Elite Outliers)")
    plt.xlabel("Current Value (€)")
    plt.show()


def describe_target(df: pd.DataFrame) -> pd.Series:
    """Print descriptive statistics (median, mean, max, skew) of the target."""
    target_summary = df["current_value"].describe()

    print("Resumo da variável alvo:")
    print(target_summary)

    print("\nMediana:", df["current_value"].median())
    print("Média:", df["current_value"].mean())
    print("Máximo:", df["current_value"].max())
    print("Assimetria:", df["current_value"].skew())

    return target_summary


def plot_age_vs_value(df: pd.DataFrame) -> None:
    """Scatterplot of player age vs market value."""
    sns.scatterplot(x=df["age"], y=df["current_value"])
    plt.title("Player Age vs Market Value")
    plt.show()


def group_position(pos) -> str:
    """Map a specific playing position string to a general category."""
    pos = str(pos).lower()

    if "goalkeeper" in pos:
        return "Goalkeeper"
    elif "back" in pos or "defender" in pos:
        return "Defender"
    elif "midfield" in pos:
        return "Midfielder"
    elif "forward" in pos or "winger" in pos or "striker" in pos:
        return "Forward"
    else:
        return "Other"


def add_general_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'general_position' column derived from 'position' and print counts."""
    df["general_position"] = df["position"].apply(group_position)
    print(df["general_position"].value_counts())
    return df


def plot_position_distribution(df: pd.DataFrame) -> None:
    """Countplot of players by general position."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="general_position", order=df["general_position"].value_counts().index)
    plt.title("Distribuição dos jogadores por posição geral")
    plt.xlabel("Posição")
    plt.ylabel("Quantidade de jogadores")
    plt.show()


def plot_goals_vs_value_by_position(df: pd.DataFrame) -> None:
    """Scatterplot of goals vs market value (log scale), colored by position."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=df,
        x="goals",
        y="current_value",
        hue="general_position",
        palette=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        alpha=0.5,
        s=50,
    )

    plt.yscale("log")

    plt.title("Positional Bias: Goals vs. Market Value (Log Scale)", fontsize=16, fontweight="bold")
    plt.xlabel("Goals per 90 Minutes", fontsize=12)
    plt.ylabel("Current Market Value (€) - Log Scale", fontsize=12)
    plt.legend(title="General Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def goal_outliers(df: pd.DataFrame, goals_threshold: int = 3) -> pd.DataFrame:
    """Return players with an unrealistic number of goals for manual inspection."""
    outliers = df[df["goals"] > goals_threshold].sort_values(by="goals", ascending=False)
    return outliers[["name", "general_position", "goals", "minutes played", "current_value"]].head(10)


def low_minutes_analysis(df: pd.DataFrame, threshold: int = 270) -> float:
    """Print/return the count and % of players with low playing time."""
    low_minutes_count = (df["minutes played"] < threshold).sum()
    low_minutes_percent = low_minutes_count / len(df) * 100

    print("Jogadores com menos de 270 minutos:", low_minutes_count)
    print(f"Percentual do dataset: {low_minutes_percent:.2f}%")

    return low_minutes_percent


def plot_minutes_played_distribution(df: pd.DataFrame, threshold: int = 270) -> None:
    """Histogram of minutes played with a vertical line at the cutoff threshold."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["minutes played"], bins=50)
    plt.axvline(threshold, linestyle="--")
    plt.title("Distribuição de minutos jogados")
    plt.xlabel("Minutos jogados")
    plt.ylabel("Quantidade de jogadores")
    plt.show()


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full EDA sequence in the same order as the original notebook.

    Returns the DataFrame with the 'general_position' column added, since
    downstream steps (e.g. the goals-vs-value plot) depend on it.
    """
    basic_info(df)
    plot_correlation_matrix(df)
    peak_value_percentage(df)

    plot_target_distribution(df)
    plot_target_boxplot(df)
    df.describe()
    describe_target(df)

    plot_age_vs_value(df)

    df = add_general_position(df)
    plot_position_distribution(df)
    plot_goals_vs_value_by_position(df)

    print(goal_outliers(df))
    low_minutes_analysis(df)
    plot_minutes_played_distribution(df)

    return df