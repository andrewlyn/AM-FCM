import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "./.mpl-cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend

import experiment_common as ec

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WAVEFORM = "ricker"
NOISE_TYPE = "WGN"
REAL_NOISE_CSV = None

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = list(range(-10, 7, 2))

REPEATS = 200

FEATURE_SETS = {

    "M": ("M",),
    "P": ("P",),
    "Std": ("Std",),

    "M+P": ("M", "P"),
    "M+Std": ("M", "Std"),

    "P+Std": ("P", "Std"),

    "M+P+Std": ("M", "P", "Std"),

}
N_JOBS = 32
BASE_SEED = 20260818
OUTPUT_DIR = Path("results_feature_combinations_global_snr")

def evaluate_one(frequency, snr_db, repeat_id, seed, noise_bank=None):
    rng = np.random.default_rng(seed)
    noisy, clean, metadata = ec.simulate(
        WAVEFORM, frequency, snr_db, noise_type=NOISE_TYPE, noise_bank=noise_bank, rng=rng
    )

    enhanced, kept_idx, _ = ec.cwt_hos_icwt(noisy)
    true_arrival = int(metadata["true_arrival"])
    rows = []

    for feature_name, feature_names in FEATURE_SETS.items():
        if kept_idx.size == 0:
            pick = ec.normalize_pick_result(ec.adaptive_fcm(None))
        else:
            features = ec.build_feature_matrix(enhanced, feature_names, ec.WINDOW_SIZE)
            pick = ec.normalize_pick_result(ec.adaptive_fcm(features))

        predicted = pick["predicted_arrival"]
        if np.isfinite(predicted):
            error_ms = (float(predicted) - true_arrival) * 1000.0 / ec.FS
            abs_error_ms = abs(error_ms)
        else:
            error_ms = np.nan
            abs_error_ms = np.nan

        rows.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "repeat_id": int(repeat_id),
            "feature_set": feature_name,
            "true_arrival": true_arrival,
            "predicted_arrival": predicted,
            "error_ms": error_ms,
            "abs_error_ms": abs_error_ms,
            "le2ms": bool(np.isfinite(abs_error_ms) and abs_error_ms <= 2.0),
            "le5ms": bool(np.isfinite(abs_error_ms) and abs_error_ms <= 5.0),
            "le10ms": bool(np.isfinite(abs_error_ms) and abs_error_ms <= 10.0),
            "reliable": bool(pick["success"]),
            "cluster_num": pick["cluster_num"],
            "valid_peak_count": pick["valid_peak_count"],
            "peak_position": pick["peak_position"],
            "stop_reason": pick["stop_reason"],
            "hos_scale_count": int(kept_idx.size),
        })

    return rows

def summarize_results(raw):
    rows = []
    grouped = raw.groupby(["frequency_hz", "snr_db", "feature_set"], sort=True)

    for (frequency, snr_db, feature_set), group in grouped:
        errors = group["error_ms"].to_numpy(dtype=float)
        abs_errors = group["abs_error_ms"].to_numpy(dtype=float)
        valid = np.isfinite(errors)

        rows.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "feature_set": feature_set,
            "repeats": int(len(group)),
            "reliable_count": int(group["reliable"].sum()),
            "reliable_rate_pct": 100.0 * float(group["reliable"].mean()),
            "le2_count": int(group["le2ms"].sum()),
            "le2_pct": 100.0 * float(group["le2ms"].mean()),
            "le5_count": int(group["le5ms"].sum()),
            "le5_pct": 100.0 * float(group["le5ms"].mean()),
            "le10_count": int(group["le10ms"].sum()),
            "le10_pct": 100.0 * float(group["le10ms"].mean()),
            "mae_ms": float(np.nanmean(abs_errors)) if np.any(valid) else np.nan,
            "median_ae_ms": float(np.nanmedian(abs_errors)) if np.any(valid) else np.nan,
            "rmse_ms": float(np.sqrt(np.nanmean(errors ** 2))) if np.any(valid) else np.nan,
            "signed_error_ms": float(np.nanmean(errors)) if np.any(valid) else np.nan,
        })

    return pd.DataFrame(rows)

def make_wide_table(summary):
    wide = summary.pivot(
        index=["frequency_hz", "snr_db"],
        columns="feature_set",
        values=["le2_pct", "le5_pct", "le10_pct", "reliable_rate_pct"],
    )
    wide.columns = [f"{feature}_{metric}" for metric, feature in wide.columns]
    return wide.reset_index()

def plot_frequency_curves(summary, frequency, save_path):
    subset = summary[summary["frequency_hz"] == frequency]
    metrics = [
        ("le2_pct", "|Error| <= 2 ms"),
        ("le5_pct", "|Error| <= 5 ms"),
        ("le10_pct", "|Error| <= 10 ms"),
    ]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True, constrained_layout=True)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for marker, feature_name in zip(markers, FEATURE_SETS):
            curve = subset[subset["feature_set"] == feature_name].sort_values("snr_db")
            ax.plot(curve["snr_db"], curve[metric], marker=marker, lw=1.5, ms=5, label=feature_name)

        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 102)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Global SNR (dB)")
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(f"CWT + calibrated-HOS + iCWT, f = {frequency} Hz, n = {REPEATS}")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_metric_overview(summary, metric, title, save_path):
    fig, axes = plt.subplots(len(FREQUENCIES), 1, figsize=(8.5, 3.1 * len(FREQUENCIES)),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    markers = ["o", "s", "^", "D"]

    for ax, frequency in zip(axes, FREQUENCIES):
        subset = summary[summary["frequency_hz"] == frequency]
        for marker, feature_name in zip(markers, FEATURE_SETS):
            curve = subset[subset["feature_set"] == feature_name].sort_values("snr_db")
            ax.plot(curve["snr_db"], curve[metric], marker=marker, lw=1.4, ms=4, label=feature_name)

        ax.set_title(f"{frequency} Hz")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Global SNR (dB)")
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(title)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if NOISE_TYPE == "real":
        if REAL_NOISE_CSV is None:
            raise ValueError("NOISE_TYPE='real' requires REAL_NOISE_CSV to be set.")
        noise_bank = ec.load_real_noise_csv(REAL_NOISE_CSV)
    else:
        noise_bank = None

    tasks = [(frequency, snr_db, repeat_id)
             for frequency in FREQUENCIES
             for snr_db in SNR_VALUES
             for repeat_id in range(REPEATS)]

    seed_sequence = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_sequence.spawn(len(tasks))
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seeds]

    print(f"Number of dominant frequencies: {len(FREQUENCIES)}")
    print(f"Number of SNR values: {len(SNR_VALUES)}")
    print(f"Repeats per condition: {REPEATS}")
    print(f"Total realizations: {len(tasks)}")
    print(f"Number of feature combinations: {len(FEATURE_SETS)}")
    print(f"Final result rows: {len(tasks) * len(FEATURE_SETS)}")
    print(f"Parallel workers: {N_JOBS}")
    print("Starting computation...")

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=N_JOBS,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(frequency, snr_db, repeat_id, seed, noise_bank)
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )

    raw = pd.DataFrame([row for block in results for row in block])
    summary = summarize_results(raw)
    wide = make_wide_table(summary)

    raw_path = OUTPUT_DIR / "feature_combination_raw_results.csv"
    summary_path = OUTPUT_DIR / "feature_combination_summary.csv"
    wide_path = OUTPUT_DIR / "feature_combination_summary_wide.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    for frequency in FREQUENCIES:
        plot_frequency_curves(
            summary,
            frequency,
            OUTPUT_DIR / f"feature_curves_{frequency}Hz.png",
        )

    plot_metric_overview(
        summary, "le2_pct", "|Error| <= 2 ms", OUTPUT_DIR / "overview_le2ms.png"
    )
    plot_metric_overview(
        summary, "le5_pct", "|Error| <= 5 ms", OUTPUT_DIR / "overview_le5ms.png"
    )
    plot_metric_overview(
        summary, "le10_pct", "|Error| <= 10 ms", OUTPUT_DIR / "overview_le10ms.png"
    )

    print("\nComputation complete。")
    print(f"Raw results: {raw_path.resolve()}")
    print(f"Summary results: {summary_path.resolve()}")
    print(f"Wide-format results: {wide_path.resolve()}")
    print("\nAverage performance by feature set:")
    print(
        summary.groupby("feature_set")[["le2_pct", "le5_pct", "le10_pct", "reliable_rate_pct"]]
        .mean()
        .round(2)
        .to_string()
    )

if __name__ == "__main__":
    main()
