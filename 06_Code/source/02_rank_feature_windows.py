from pathlib import Path
import numpy as np
import pandas as pd

INPUT_FILE = Path(
    "results_window_feature_relevance/window_feature_mean_by_condition.csv"
)
OUTPUT_DIR = Path("results_window_feature_relevance")
SELECTED_FEATURES = ("P", "M", "Std")

def calculate_window_scores(data):
    required = {
        "window_size", "frequency_hz", "snr_db", "feature",
        "pearson_mean", "spearman_mean", "mi_mean",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    selected = data[data["feature"].isin(SELECTED_FEATURES)].copy()
    if selected.empty:
        raise ValueError("No P/M/Std data found.")

    feature_window = selected.groupby(
        ["window_size", "feature"], as_index=False, observed=True
    ).agg(
        condition_count=("snr_db", "count"),
        pearson_score=("pearson_mean", "mean"),
        spearman_score=("spearman_mean", "mean"),
        mi_score=("mi_mean", "mean"),
    )

    window = feature_window.groupby(
        "window_size", as_index=False, observed=True
    ).agg(
        feature_count=("feature", "nunique"),
        pearson_score=("pearson_score", "mean"),
        spearman_score=("spearman_score", "mean"),
        mi_score=("mi_score", "mean"),
    )

    window["rank_pearson"] = window["pearson_score"].rank(
        ascending=False, method="average"
    )
    window["rank_spearman"] = window["spearman_score"].rank(
        ascending=False, method="average"
    )
    window["rank_mi"] = window["mi_score"].rank(
        ascending=False, method="average"
    )

    window["R_mean_rank"] = (
        window["rank_pearson"]
        + window["rank_spearman"]
        + window["rank_mi"]
    ) / 3.0

    window["overall_rank"] = window["R_mean_rank"].rank(
        ascending=True, method="min"
    ).astype(int)

    return feature_window, window.sort_values(
        ["R_mean_rank", "window_size"]
    ).reset_index(drop=True)

def validate_conditions(data):
    expected = data["frequency_hz"].nunique() * data["snr_db"].nunique()
    check = data[data["feature"].isin(SELECTED_FEATURES)].groupby(
        ["window_size", "feature"], observed=True
    ).size()

    bad = check[check != expected]
    if not bad.empty:
        print("Warning: the following window-feature condition counts are incomplete", expected)
        print(bad)
    else:
        print(f"Complete conditions: each window-feature contains {expected}  frequency-SNR conditions.")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run program 1 first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)
    validate_conditions(data)

    feature_window, window_rank = calculate_window_scores(data)

    feature_window.to_csv(
        OUTPUT_DIR / "P_M_Std_scores_by_window.csv",
        index=False, encoding="utf-8-sig"
    )
    window_rank.to_csv(
        OUTPUT_DIR / "window_overall_rank_P_M_Std.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\nWindow ranking based on P/M/Std")
    print("-" * 88)
    print(
        window_rank[
            [
                "window_size",
                "pearson_score", "rank_pearson",
                "spearman_score", "rank_spearman",
                "mi_score", "rank_mi",
                "R_mean_rank", "overall_rank",
            ]
        ].round(4).to_string(index=False)
    )

    best = window_rank.iloc[0]
    print("-" * 88)
    print(
        f"Best overall window: {int(best['window_size'])}, "
        f"R(w)={best['R_mean_rank']:.4f}"
    )

if __name__ == "__main__":
    main()
