import os, sys, time
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
from fcmeans import FCM
from scipy.signal import find_peaks

import experiment_common as ec

WAVEFORM = "ricker"
NOISE_TYPE = "WGN"
REAL_NOISE_CSV = None

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = list(range(-10, 7, 2))
REPEATS = 200

FEATURE_NAMES = ("M", "Std")
FIXED_CLUSTERS = [2, 3, 6]
METHODS = ["C=2", "C=3", "C=6", "AM-FCM"]

N_JOBS = 32
BASE_SEED = 20260915
OUTPUT_DIR = Path("results_cluster_number_comparison")

def _empty_fixed_result(cluster_num, stop_reason="not_started", message=""):
    return {
        "arrival": None,
        "cluster_num": int(cluster_num),
        "stable_count": 0,
        "valid_peak_count": 0,
        "valid_peaks": np.empty(0, dtype=int),
        "peak": None,
        "peak_membership": np.nan,
        "onset_level": np.nan,
        "onset_baseline": np.nan,
        "stop_reason": stop_reason,
        "diagnostic_message": message,
        "success": False,
    }

def fixed_fcm(features, cluster_num):

    if features is None:
        return _empty_fixed_result(cluster_num, "empty_feature")

    result = _empty_fixed_result(cluster_num)

    try:
        model = FCM(
            n_clusters=int(cluster_num),
            max_iter=ec.FCM_MAX_ITER,
            m=ec.FCM_M,
            error=ec.FCM_ERROR,
            random_state=ec.FCM_RANDOM_STATE,
        )
        model.fit(features)

        centers = np.asarray(model.centers)
        membership = np.asarray(model.u)

        if centers.ndim != 2 or membership.ndim != 2 or membership.shape[0] != features.shape[0]:
            raise ValueError("Invalid FCM output dimensions.")
    except Exception as error:
        return _empty_fixed_result(cluster_num, "fcm_failed", str(error))

    signal_cluster = int(np.argmax(centers.mean(axis=1)))
    signal_membership = membership[:, signal_cluster]

    peaks, properties = find_peaks(
        signal_membership,
        height=ec.PEAK_HEIGHT,
        prominence=ec.PEAK_PROMINENCE,
        distance=ec.PEAK_DISTANCE,
        width=ec.PEAK_WIDTH,
    )

    valid_peaks, _ = ec.merge_region_peaks(
        signal_membership,
        peaks,
        properties["prominences"],
    )

    valid_peak_count = int(valid_peaks.size)
    earliest_peak = int(valid_peaks[0]) if valid_peak_count > 0 else None

    result.update({
        "valid_peak_count": valid_peak_count,
        "valid_peaks": valid_peaks.copy(),
        "peak": earliest_peak,
        "peak_membership": float(signal_membership[earliest_peak]) if earliest_peak is not None else np.nan,
    })

    if valid_peak_count == 0:
        result["stop_reason"] = "no_valid_peak"
        return result

    if valid_peak_count > 1:
        result["stop_reason"] = "multiple_valid_peaks"
        return result

    arrival, onset_level, onset_baseline = ec.backward_pick_relative(
        signal_membership, earliest_peak
    )

    result.update({
        "arrival": arrival,
        "onset_level": float(onset_level),
        "onset_baseline": float(onset_baseline),
    })

    if arrival is None:
        result["stop_reason"] = "unique_peak_no_onset_crossing"
        return result

    result["stop_reason"] = "unique_valid_peak"
    result["success"] = True
    return result

def build_result_row(frequency, snr_db, repeat_id, method, true_arrival, pick, kept_idx):
    predicted = pick["predicted_arrival"]

    if bool(pick["success"]) and np.isfinite(predicted):
        error_ms = (float(predicted) - float(true_arrival)) * 1000.0 / ec.FS
        abs_error_ms = abs(error_ms)
    else:
        error_ms = np.nan
        abs_error_ms = np.nan

    return {
        "frequency_hz": int(frequency),
        "snr_db": float(snr_db),
        "repeat_id": int(repeat_id),
        "method": method,
        "true_arrival": int(true_arrival),
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
    }

def evaluate_one(frequency, snr_db, repeat_id, seed, noise_bank=None):

    rng = np.random.default_rng(seed)

    noisy, clean, metadata = ec.simulate(
        WAVEFORM, frequency, snr_db,
        noise_type=NOISE_TYPE, noise_bank=noise_bank, rng=rng
    )

    enhanced, kept_idx, _ = ec.cwt_hos_icwt(noisy)
    true_arrival = int(metadata["true_arrival"])
    rows = []

    if kept_idx.size == 0:
        features = None
    else:
        features = ec.build_feature_matrix(enhanced, FEATURE_NAMES, ec.WINDOW_SIZE)

    for cluster_num in FIXED_CLUSTERS:
        pick = ec.normalize_pick_result(fixed_fcm(features, cluster_num))
        rows.append(
            build_result_row(
                frequency, snr_db, repeat_id,
                f"C={cluster_num}", true_arrival, pick, kept_idx
            )
        )

    pick = ec.normalize_pick_result(ec.adaptive_fcm(features))
    rows.append(
        build_result_row(
            frequency, snr_db, repeat_id,
            "AM-FCM", true_arrival, pick, kept_idx
        )
    )

    return rows

def summarize_results(raw):
    rows = []
    grouped = raw.groupby(["frequency_hz", "snr_db", "method"], sort=True)

    for (frequency, snr_db, method), group in grouped:
        errors = group["error_ms"].to_numpy(dtype=float)
        abs_errors = group["abs_error_ms"].to_numpy(dtype=float)
        valid = np.isfinite(errors)

        rows.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "method": method,
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

def make_macro_average(summary):
    metrics = [
        "le2_pct", "le5_pct", "le10_pct", "reliable_rate_pct",
        "mae_ms", "median_ae_ms", "rmse_ms", "signed_error_ms",
    ]

    return (
        summary
        .groupby(["snr_db", "method"], as_index=False)[metrics]
        .mean()
    )

def make_wide_comparison(macro):
    value_cols = [
        "le2_pct", "le5_pct", "le10_pct",
        "reliable_rate_pct", "mae_ms", "signed_error_ms",
    ]

    wide = macro.pivot(
        index="snr_db",
        columns="method",
        values=value_cols,
    )
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    return wide.reset_index()

def summarize_amfcm_cluster_distribution(raw):
    am = raw[
        (raw["method"] == "AM-FCM")
        & (raw["reliable"])
        & np.isfinite(raw["cluster_num"])
    ].copy()

    if am.empty:
        return pd.DataFrame()

    counts = (
        am.groupby(["snr_db", "cluster_num"])
        .size()
        .reset_index(name="count")
    )

    totals = counts.groupby("snr_db")["count"].transform("sum")
    counts["percentage"] = 100.0 * counts["count"] / totals
    return counts

def plot_macro_curves(macro, save_path):
    metrics = [
        ("le2_pct", "|Error| ≤ 2 ms"),
        ("le5_pct", "|Error| ≤ 5 ms"),
        ("le10_pct", "|Error| ≤ 10 ms"),
    ]

    markers = {
        "C=2": "o",
        "C=3": "s",
        "C=6": "^",
        "AM-FCM": "D",
    }

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8),
        sharey=True, constrained_layout=True
    )

    for ax, (metric, title) in zip(axes, metrics):
        for method in METHODS:
            curve = macro[macro["method"] == method].sort_values("snr_db")
            ax.plot(
                curve["snr_db"], curve[metric],
                marker=markers[method],
                linewidth=1.6, markersize=5,
                label=method
            )

        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Picking accuracy (%)")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Fixed-cluster FCM vs AM-FCM (frequency-macro-averaged)")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_frequency_metric(summary, metric, frequencies, save_path):
    markers = {
        "C=2": "o",
        "C=3": "s",
        "C=6": "^",
        "AM-FCM": "D",
    }

    fig, axes = plt.subplots(
        len(frequencies), 1,
        figsize=(8.5, 3.0 * len(frequencies)),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, frequency in zip(axes, frequencies):
        subset = summary[summary["frequency_hz"] == frequency]

        for method in METHODS:
            curve = subset[subset["method"] == method].sort_values("snr_db")
            ax.plot(
                curve["snr_db"], curve[metric],
                marker=markers[method],
                linewidth=1.4, markersize=4,
                label=method
            )

        ax.set_title(f"{frequency} Hz")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("SNR (dB)")
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    quick = "--quick" in sys.argv[1:]

    if quick:
        frequencies = [100]
        snr_values = [-10, -6, 0]
        repeats = 20
    else:
        frequencies = FREQUENCIES
        snr_values = SNR_VALUES
        repeats = REPEATS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if NOISE_TYPE == "real":
        if REAL_NOISE_CSV is None:
            raise ValueError("NOISE_TYPE='real' requires REAL_NOISE_CSV to be set.")
        noise_bank = ec.load_real_noise_csv(REAL_NOISE_CSV)
    else:
        noise_bank = None

    tasks = [
        (frequency, snr_db, repeat_id)
        for frequency in frequencies
        for snr_db in snr_values
        for repeat_id in range(repeats)
    ]

    seed_sequence = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_sequence.spawn(len(tasks))
    seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in child_seeds
    ]

    print("=" * 78)
    print("Fixed FCM C=2/3/6 vs AM-FCM")
    print("=" * 78)
    print(f"Frequencies : {frequencies}")
    print(f"SNR         : {snr_values}")
    print(f"Repeats     : {repeats}")
    print(f"Features    : {FEATURE_NAMES}")
    print(f"Noise       : {NOISE_TYPE}")
    print(f"Methods     : {METHODS}")
    print(f"Jobs        : {N_JOBS}")
    print(f"Realizations: {len(tasks)}")
    print(f"Total method records: {len(tasks) * len(METHODS)}")
    print("-" * 78)

    t0 = time.time()

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=N_JOBS,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(
                frequency, snr_db, repeat_id, seed, noise_bank
            )
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )

    elapsed = time.time() - t0

    raw = pd.DataFrame([
        row
        for block in results
        for row in block
    ])

    summary = summarize_results(raw)
    macro = make_macro_average(summary)
    wide = make_wide_comparison(macro)
    cluster_dist = summarize_amfcm_cluster_distribution(raw)

    raw_path = OUTPUT_DIR / "cluster_comparison_raw.csv"
    summary_path = OUTPUT_DIR / "cluster_comparison_summary_by_frequency.csv"
    macro_path = OUTPUT_DIR / "cluster_comparison_macroavg_by_snr.csv"
    wide_path = OUTPUT_DIR / "cluster_comparison_macroavg_wide.csv"
    cluster_path = OUTPUT_DIR / "amfcm_cluster_distribution.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    if not cluster_dist.empty:
        cluster_dist.to_csv(cluster_path, index=False, encoding="utf-8-sig")

    plot_macro_curves(
        macro,
        OUTPUT_DIR / "cluster_comparison_macroavg.png"
    )

    if not quick:
        plot_frequency_metric(
            summary, "le2_pct", frequencies,
            OUTPUT_DIR / "cluster_comparison_le2_by_frequency.png"
        )
        plot_frequency_metric(
            summary, "le5_pct", frequencies,
            OUTPUT_DIR / "cluster_comparison_le5_by_frequency.png"
        )
        plot_frequency_metric(
            summary, "le10_pct", frequencies,
            OUTPUT_DIR / "cluster_comparison_le10_by_frequency.png"
        )

    print(f"\nComputation complete in {elapsed:.1f} s")

    display_cols = [
        "snr_db", "method",
        "le2_pct", "le5_pct", "le10_pct",
        "reliable_rate_pct",
        "mae_ms", "signed_error_ms",
    ]

    print("\nFrequency-averaged results:")
    print(
        macro[display_cols]
        .round(2)
        .to_string(index=False)
    )

    print("\nResult files:")
    print(raw_path.resolve())
    print(summary_path.resolve())
    print(macro_path.resolve())
    print(wide_path.resolve())
    if not cluster_dist.empty:
        print(cluster_path.resolve())

if __name__ == "__main__":
    main()
