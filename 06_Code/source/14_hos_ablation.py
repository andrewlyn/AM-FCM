import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "./.mpl-cache")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

_BASE_DIR = Path(__file__).resolve().parent
import experiment_common as ec

WAVEFORM = "ricker"
NOISE_TYPE = "WGN"
FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6, -4, -2, 0, 2, 4, 6]
N_REPEAT = 200
N_JOBS = 6

BASE_SEED = 20260818
FEATURE_SET = ("M", "Std")
WINDOW_SIZE = ec.WINDOW_SIZE

OUTPUT_DIR = _BASE_DIR / "results_hos_ablation_mstd_paired"

def empty_pick(reason):

    return {
        "predicted_arrival": np.nan,
        "cluster_num": np.nan,
        "valid_peak_count": 0,
        "peak_position": np.nan,
        "stop_reason": reason,
        "success": False,
    }

def run_picker(enhanced):

    features = ec.build_feature_matrix(enhanced, FEATURE_SET, WINDOW_SIZE)
    if features is None:
        return empty_pick("empty_feature")
    return ec.normalize_pick_result(ec.adaptive_fcm(features))

def preprocess_pair(noisy):

    coeffs, freqs = ec.cwt_morlet_pywt(noisy)
    scales = ec.get_analysis_scales(freqs)

    filtered, kept_idx = ec.hos_preprocess_cwt(coeffs, freqs)
    enhanced_hos = ec.inverse_cwt(filtered, scales)

    enhanced_no_hos = ec.inverse_cwt(coeffs, scales)

    return enhanced_hos, enhanced_no_hos, int(kept_idx.size), int(len(freqs))

def evaluate_one(frequency, snr_db, repeat_id, seed):
    rng = np.random.default_rng(seed)
    noisy, clean, metadata = ec.simulate(
        WAVEFORM, frequency, snr_db, noise_type=NOISE_TYPE, rng=rng
    )
    true_arrival = int(metadata["true_arrival"])

    enhanced_hos, enhanced_no_hos, kept_count, total_scale_count = preprocess_pair(noisy)
    branches = {
        "with_HOS": (enhanced_hos, kept_count),
        "without_HOS": (enhanced_no_hos, total_scale_count),
    }

    rows = []
    for variant, (enhanced, used_scale_count) in branches.items():
        pick = run_picker(enhanced)
        pred = pick["predicted_arrival"]

        if np.isfinite(pred):
            error_ms = (float(pred) - true_arrival) * 1000.0 / ec.FS
            abs_error_ms = abs(error_ms)
        else:
            error_ms = np.nan
            abs_error_ms = np.nan

        rows.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "repeat_id": int(repeat_id),
            "variant": variant,
            "true_arrival": true_arrival,
            "predicted_arrival": pred,
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
            "used_scale_count": int(used_scale_count),
            "hos_kept_scale_count": int(kept_count),
            "total_scale_count": int(total_scale_count),
        })
    return rows

def summarize_results(raw):
    rows = []
    grouped = raw.groupby(["variant", "frequency_hz", "snr_db"], sort=True)

    for (variant, frequency, snr_db), group in grouped:
        errors = group["error_ms"].to_numpy(dtype=float)
        abs_errors = group["abs_error_ms"].to_numpy(dtype=float)
        valid = np.isfinite(errors)

        rows.append({
            "variant": variant,
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
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
            "mean_used_scale_count": float(group["used_scale_count"].mean()),
        })

    return pd.DataFrame(rows)

def aggregate_across_frequency(summary):

    metrics = [
        "reliable_rate_pct", "le2_pct", "le5_pct", "le10_pct",
        "mae_ms", "median_ae_ms", "rmse_ms", "signed_error_ms"
    ]
    return (
        summary.groupby(["variant", "snr_db"], as_index=False)[metrics]
        .mean()
        .sort_values(["variant", "snr_db"])
        .reset_index(drop=True)
    )

def make_comparison_table(agg):

    metrics = ["reliable_rate_pct", "le2_pct", "le5_pct", "le10_pct"]
    with_hos = agg[agg["variant"] == "with_HOS"][["snr_db"] + metrics].copy()
    no_hos = agg[agg["variant"] == "without_HOS"][["snr_db"] + metrics].copy()

    compare = with_hos.merge(no_hos, on="snr_db", suffixes=("_with_HOS", "_without_HOS"))
    for metric in metrics:
        compare[f"delta_{metric}"] = (
            compare[f"{metric}_with_HOS"] - compare[f"{metric}_without_HOS"]
        )
    return compare

def make_paired_error_table(raw):

    keys = ["frequency_hz", "snr_db", "repeat_id"]
    paired = raw.pivot(index=keys, columns="variant", values="abs_error_ms").reset_index()
    paired.columns.name = None

    if "with_HOS" in paired.columns and "without_HOS" in paired.columns:
        paired["error_reduction_by_HOS_ms"] = paired["without_HOS"] - paired["with_HOS"]
    else:
        paired["error_reduction_by_HOS_ms"] = np.nan
    return paired

def plot_comparison(agg, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("le2_pct", "|Error| <= 2 ms"),
        ("le5_pct", "|Error| <= 5 ms"),
        ("le10_pct", "|Error| <= 10 ms"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for variant, marker, linestyle in [
            ("with_HOS", "o", "-"),
            ("without_HOS", "s", "--"),
        ]:
            curve = agg[agg["variant"] == variant].sort_values("snr_db")
            ax.plot(curve["snr_db"], curve[metric], marker=marker,
                    linestyle=linestyle, linewidth=1.6, markersize=5, label=variant)

        ax.set_title(title)
        ax.set_xlabel("Global SNR (dB)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Cross-frequency averaged hit rate (%)")
    axes[0].legend()
    fig.suptitle("HOS ablation: M+Std + AM-FCM")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    quick = "--quick" in sys.argv[1:]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if quick:
        freq_use = FREQUENCIES[:1]
        snr_use = [-10, -4, 2]
        n_repeat = 20
    else:
        freq_use = FREQUENCIES
        snr_use = SNR_VALUES
        n_repeat = N_REPEAT

    tasks = [
        (frequency, snr_db, repeat_id)
        for frequency in freq_use
        for snr_db in snr_use
        for repeat_id in range(n_repeat)
    ]

    seed_sequence = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_sequence.spawn(len(tasks))
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seeds]

    print("=" * 80)
    print("Strict paired HOS ablation: M+Std + AM-FCM")
    print("With HOS    : CWT -> HOS -> iCWT -> M+Std -> AM-FCM")
    print("Without HOS : CWT -------> iCWT -> M+Std -> AM-FCM")
    print("=" * 80)
    print(f"Frequencies : {freq_use}")
    print(f"SNR values  : {snr_use}")
    print(f"Repeats     : {n_repeat}")
    print(f"Features    : {FEATURE_SET}, window={WINDOW_SIZE}")
    print(f"Reference   : AIC(clean)")
    print(f"SNR method  : inherited from experiment_common.simulate")
    print(f"Parallel    : {N_JOBS} jobs")
    print(f"Realizations: {len(tasks)}; paired evaluations: {2 * len(tasks)}")
    print("-" * 80)

    t0 = time.time()
    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=N_JOBS,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(frequency, snr_db, repeat_id, seed)
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )
    print(f"\nComputation finished in {time.time() - t0:.1f} s.")

    raw = pd.DataFrame([row for block in results for row in block])
    summary = summarize_results(raw)
    agg = aggregate_across_frequency(summary)
    compare = make_comparison_table(agg)
    paired = make_paired_error_table(raw)

    raw_path = OUTPUT_DIR / "hos_ablation_raw_paired.csv"
    summary_path = OUTPUT_DIR / "hos_ablation_summary_by_frequency_snr.csv"
    agg_path = OUTPUT_DIR / "hos_ablation_macroavg_by_snr.csv"
    compare_path = OUTPUT_DIR / "hos_ablation_compare_by_snr.csv"
    paired_path = OUTPUT_DIR / "hos_ablation_paired_abs_error.csv"
    fig_path = OUTPUT_DIR / "hos_ablation_hit_rates.png"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    plot_comparison(agg, fig_path)

    print("\nCross-frequency averaged hit rates (%):")
    print(
        agg[["variant", "snr_db", "le2_pct", "le5_pct", "le10_pct", "reliable_rate_pct"]]
        .round(2)
        .to_string(index=False)
    )

    print("\nWith-HOS minus Without-HOS (percentage points):")
    delta_cols = ["snr_db", "delta_le2_pct", "delta_le5_pct", "delta_le10_pct", "delta_reliable_rate_pct"]
    print(compare[delta_cols].round(2).to_string(index=False))

    print("\nSaved files:")
    for path in [raw_path, summary_path, agg_path, compare_path, paired_path, fig_path]:
        print(path.resolve())

if __name__ == "__main__":
    main()
