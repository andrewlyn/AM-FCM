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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

import experiment_common as ec

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6, -4, -2, 0, 2, 4, 6]
N_REPEAT = 200
N_JOBS = min(32, os.cpu_count() or 1)
BASE_SEED = 20260915
FEATURE_SET = ("M", "Std")

SCRIPT_DIR = Path(__file__).resolve().parent

def _safe_pick(noisy):

    try:
        return ec.pick_signal(
            noisy,
            feature_names=FEATURE_SET,
            preprocessing=getattr(ec, "PREPROCESSING", "icwt"),
        )
    except TypeError:

        return ec.pick_signal(noisy, feature_names=FEATURE_SET)

def _one_row(condition_name, frequency, snr_db, repeat_id, noisy, metadata):
    result = _safe_pick(noisy)
    pred = result.get("predicted_arrival", np.nan)
    success = bool(result.get("success", False))
    true_arrival = int(metadata["true_arrival"])

    if success and pred is not None and np.isfinite(pred):
        error_ms = (float(pred) - true_arrival) * 1000.0 / float(ec.FS)
        abs_error_ms = abs(error_ms)
    else:
        error_ms = np.nan
        abs_error_ms = np.nan

    return {
        "condition": condition_name,
        "frequency_hz": int(frequency),
        "snr_db": float(snr_db),
        "repeat": int(repeat_id),
        "true_arrival": true_arrival,
        "predicted_arrival": pred,
        "success": success,
        "cluster_num": result.get("cluster_num", np.nan),
        "stop_reason": result.get("stop_reason", ""),
        "error_ms": error_ms,
        "abs_error_ms": abs_error_ms,
        "hit_le2": bool(success and abs_error_ms <= 2.0),
        "hit_le5": bool(success and abs_error_ms <= 5.0),
        "hit_le10": bool(success and abs_error_ms <= 10.0),
        "target_snr_db": metadata.get("target_snr_db", snr_db),
        "snr_definition": metadata.get("snr_definition", ""),
        "noise_source_id": metadata.get("noise_source_id", -1),
        "noise_start_sample": metadata.get("noise_start_sample", -1),
    }

def summarize(group):
    n = len(group)
    success = group["success"].fillna(False).astype(bool)
    err = pd.to_numeric(group.loc[success, "error_ms"], errors="coerce").dropna().to_numpy(float)
    ae = np.abs(err)

    return pd.Series({
        "n": int(n),
        "reliable_count": int(success.sum()),
        "reliable_rate_pct": 100.0 * success.mean() if n else 0.0,
        "le2_pct": 100.0 * group["hit_le2"].mean() if n else 0.0,
        "le5_pct": 100.0 * group["hit_le5"].mean() if n else 0.0,
        "le10_pct": 100.0 * group["hit_le10"].mean() if n else 0.0,
        "mae_ms": float(np.mean(ae)) if ae.size else np.nan,
        "median_ae_ms": float(np.median(ae)) if ae.size else np.nan,
        "rmse_ms": float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan,
        "signed_error_ms": float(np.mean(err)) if err.size else np.nan,
    })

def save_summaries(raw, output_dir):

    by_condition = (
        raw.groupby(["condition", "frequency_hz", "snr_db"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    by_snr = (
        raw.groupby(["condition", "snr_db"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_condition.to_csv(
        output_dir / "summary_by_frequency_snr.csv",
        index=False, encoding="utf-8-sig"
    )
    by_snr.to_csv(
        output_dir / "summary_cross_frequency_by_snr.csv",
        index=False, encoding="utf-8-sig"
    )

    metrics = ["le2_pct", "le5_pct", "le10_pct"]
    wide = by_snr.pivot(index="snr_db", columns="condition", values=metrics)
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(
        output_dir / "comparison_cross_frequency_by_snr.csv",
        index=False, encoding="utf-8-sig"
    )
    return by_condition, by_snr, wide

def plot_comparison(by_snr, conditions, title, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True, sharey=True)
    metrics = [
        ("le2_pct", r"$P(|err|\leq2\,ms)$"),
        ("le5_pct", r"$P(|err|\leq5\,ms)$"),
        ("le10_pct", r"$P(|err|\leq10\,ms)$"),
    ]

    for ax, (metric, panel_title) in zip(axes, metrics):
        for condition in conditions:
            sub = by_snr[by_snr["condition"] == condition].sort_values("snr_db")
            ax.plot(sub["snr_db"], sub[metric], marker="o", linewidth=1.8, label=condition)
        ax.set_title(panel_title)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Hit rate (%)")
    axes[-1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

WAVEFORM = "ricker"
NOISE_TYPES = ["WGN", "real"]
OUTPUT_DIR = SCRIPT_DIR / "results_noise_robustness_MStd"

_noise_candidates = [
    SCRIPT_DIR / "data_all.csv",
    SCRIPT_DIR.parent / "data_all.csv",
]
REAL_NOISE_FILE = next((p for p in _noise_candidates if p.exists()), _noise_candidates[0])

def evaluate_pair(frequency_index, frequency, snr_db, repeat_id, noise_bank):

    seed_seq = np.random.SeedSequence([BASE_SEED, frequency_index, repeat_id])
    seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])

    rows = []
    for noise_type in NOISE_TYPES:
        rng = np.random.default_rng(seed)
        noisy, _, metadata = ec.simulate(
            WAVEFORM, frequency, snr_db,
            noise_type=noise_type,
            noise_bank=noise_bank if noise_type == "real" else None,
            rng=rng,
        )
        label = "WGN" if noise_type == "WGN" else "Recorded background noise"
        rows.append(
            _one_row(label, frequency, snr_db, repeat_id, noisy, metadata)
        )
    return rows

def main():
    quick = "--quick" in sys.argv[1:]
    freq_use = FREQUENCIES[:1] if quick else FREQUENCIES
    snr_use = [-10, -4, 2] if quick else SNR_VALUES
    n_repeat = 20 if quick else N_REPEAT

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REAL_NOISE_FILE.exists():
        raise FileNotFoundError(
            f"data_all.csv was not found.\nPlace it at:\n"
            f"  {SCRIPT_DIR / 'data_all.csv'}\nor\n"
            f"  {SCRIPT_DIR.parent / 'data_all.csv'}"
        )
    noise_bank = ec.load_real_noise_csv(REAL_NOISE_FILE)

    print("=" * 78, flush=True)
    print("Noise robustness: WGN vs recorded background noise, waveform = Ricker", flush=True)
    print(f"Frequencies: {freq_use}", flush=True)
    print(f"SNR: {snr_use}", flush=True)
    print(f"Repeats: {n_repeat} per frequency-SNR-noise type", flush=True)
    print(f"Feature set: {FEATURE_SET}; window = {getattr(ec, 'WINDOW_SIZE', 'from common')}", flush=True)
    print(f"Real-noise file: {REAL_NOISE_FILE}", flush=True)
    print(f"Noise records: {len(noise_bank)}", flush=True)
    print(f"Parallel jobs: {N_JOBS}", flush=True)
    print("=" * 78, flush=True)

    tasks = [
        (f_idx, freq, snr, rep)
        for f_idx, freq in enumerate(freq_use)
        for snr in snr_use
        for rep in range(n_repeat)
    ]

    t0 = time.time()
    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        nested = Parallel(
            n_jobs=N_JOBS, batch_size="auto", pre_dispatch="2*n_jobs", verbose=10
        )(delayed(evaluate_pair)(*task, noise_bank) for task in tasks)

    rows = [row for pair in nested for row in pair]
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "raw_results.csv", index=False, encoding="utf-8-sig")

    by_condition, by_snr, wide = save_summaries(raw, OUTPUT_DIR)
    labels = ["WGN", "Recorded background noise"]
    plot_comparison(
        by_snr, labels,
        "Noise robustness: WGN vs recorded seismic background noise",
        OUTPUT_DIR / "noise_robustness_le2_le5_le10.png"
    )

    print(f"\nFinished in {time.time()-t0:.1f} s", flush=True)
    print("\nCross-frequency results:", flush=True)
    print(
        by_snr[
            ["condition", "snr_db", "le2_pct", "le5_pct", "le10_pct",
             "reliable_rate_pct", "mae_ms", "signed_error_ms"]
        ].round(3).to_string(index=False),
        flush=True
    )
    print(f"\nOutput: {OUTPUT_DIR.resolve()}", flush=True)

if __name__ == "__main__":
    main()
