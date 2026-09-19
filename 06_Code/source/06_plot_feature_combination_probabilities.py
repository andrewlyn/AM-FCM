from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "feature_combination_summary.csv"
OUTPUT_DIR = SCRIPT_DIR / "feature_probability_plots"

FEATURE_GROUPS = {
    "Single Features": ["M", "P", "Std"],
    "Double Features": ["M+P", "M+Std", "P+Std"],
    "Triple Features": ["M+P+Std"],
}
SELECTED_FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]
METRICS = {
    "le2_pct": "Probability ≤ 2 ms (%)",
    "le5_pct": "Probability ≤ 5 ms (%)",
    "le10_pct": "Probability ≤ 10 ms (%)",
}

def read_and_aggregate(input_file: Path, snrs: list[float] | None) -> pd.DataFrame:

    df = pd.read_csv(input_file)

    required_columns = {"frequency_hz", "snr_db", "feature_set", *METRICS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[df["feature_set"].isin(SELECTED_FEATURES)].copy()
    if snrs is not None:
        df = df[df["snr_db"].isin(snrs)].copy()

    if df.empty:
        raise ValueError("No data remain after filtering; check --snrs or feature_set.")

    aggregated = (
        df.groupby(["snr_db", "feature_set"], as_index=False, sort=True)[list(METRICS)]
        .mean()
    )
    aggregated["feature_set"] = pd.Categorical(
        aggregated["feature_set"], categories=SELECTED_FEATURES, ordered=True
    )
    return aggregated.sort_values(["snr_db", "feature_set"]).reset_index(drop=True)

def make_wide_table(aggregated: pd.DataFrame) -> pd.DataFrame:

    wide = aggregated.pivot(index="snr_db", columns="feature_set", values=list(METRICS))
    wide.columns = [f"{feature}_{metric}" for metric, feature in wide.columns]
    return wide.reset_index().sort_values("snr_db").reset_index(drop=True)

def categorize_features(feature_set: str) -> str:

    count = len(str(feature_set).split("+"))
    if count == 1:
        return "Single Features"
    if count == 2:
        return "Double Features"
    return "Triple Features"

def plot_category_metrics(
    category_name: str,
    aggregated: pd.DataFrame,
    output_dir: Path,
) -> Path:

    feature_order = FEATURE_GROUPS[category_name]
    df_category = aggregated[aggregated["feature_set"].isin(feature_order)].copy()
    df_category["feature_set"] = pd.Categorical(
        df_category["feature_set"], categories=feature_order, ordered=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=False)
    fig.suptitle(
        f"Average Probability Metrics for {category_name}",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    palette = sns.color_palette("tab10", n_colors=len(feature_order))
    snr_ticks = sorted(df_category["snr_db"].unique())

    for axis, (metric, ylabel) in zip(axes, METRICS.items()):
        sns.lineplot(
            data=df_category,
            x="snr_db",
            y=metric,
            hue="feature_set",
            hue_order=feature_order,
            palette=palette,
            marker="o",
            markersize=5,
            linewidth=1.8,
            ci=None,
            ax=axis,
        )
        axis.set_title(ylabel, fontsize=12, fontweight="bold")
        axis.set_xlabel("SNR (dB)", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_xticks(snr_ticks)
        axis.set_ylim(0, 100)
        axis.grid(True, linestyle="--", alpha=0.55)
        axis.tick_params(axis="x", rotation=45)

        legend = axis.get_legend()
        if axis is axes[-1]:
            if legend is not None:
                legend.set_title("Feature Combination")
                legend.set_bbox_to_anchor((1.02, 1))
                legend._loc = 2  # upper-left
        elif legend is not None:
            legend.remove()

    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    safe_name = category_name.replace(" ", "_")
    output_file = output_dir / f"{safe_name}_probability_metrics.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=None,
        help="Optional: plot only specified SNR values; default uses all SNR values in the CSV.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    aggregated = read_and_aggregate(args.input, args.snrs)
    wide = make_wide_table(aggregated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "feature_probability_mean_by_snr.csv"
    wide.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Aggregated rows: {len(aggregated)}")
    print(f"SNR values: {sorted(aggregated['snr_db'].unique().tolist())}")
    print(f"Feature sets: {SELECTED_FEATURES}")
    print("\nAverage probability table")
    print(wide.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved: {output_csv}")

    for category_name in FEATURE_GROUPS:
        output_file = plot_category_metrics(category_name, aggregated, args.output_dir)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
