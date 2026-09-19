from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from joblib import Parallel, delayed, parallel_backend

import benchmark_baseline_cost as _baseline_compat  # noqa: F401

if not hasattr(pywt, "frequency2scale"):
    pywt.frequency2scale = lambda wavelet, normalized_frequency: (
        pywt.central_frequency(wavelet) / np.asarray(normalized_frequency, dtype=float)
    )

import experiment_common as ec
import cluster_number_comparison as am

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6]
REPEATS = 200
N_JOBS = 6
BASE_SEED = 20260917
OUTPUT_DIR = Path(__file__).resolve().parent / "results_amfcm_cost"

def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan

def benchmark_one(frequency: int, snr_db: float, repeat_id: int, seed: int):
    rng = np.random.default_rng(int(seed))
    noisy, _, metadata = ec.simulate(
        am.WAVEFORM,
        frequency,
        snr_db,
        noise_type=am.NOISE_TYPE,
        noise_bank=None,
        rng=rng,
    )

    pipeline_start = time.perf_counter_ns()
    enhanced, kept_idx, _ = ec.cwt_hos_icwt(noisy)
    features = (
        None
        if kept_idx.size == 0
        else ec.build_feature_matrix(enhanced, am.FEATURE_NAMES, ec.WINDOW_SIZE)
    )
    preprocess_done = time.perf_counter_ns()

    picker_start = time.perf_counter_ns()
    am_pick = ec.normalize_pick_result(am.ec.adaptive_fcm(features))
    picker_done = time.perf_counter_ns()
    rows = [{
        "method": "AM-FCM",
        "final_cluster_num": _safe_float(am_pick["cluster_num"]),
        "picker_time_ms": (picker_done - picker_start) / 1_000_000.0,
        "pipeline_time_ms": (picker_done - pipeline_start) / 1_000_000.0,
        "preprocess_time_ms": (preprocess_done - pipeline_start) / 1_000_000.0,
        "predicted_arrival": _safe_float(am_pick["predicted_arrival"]),
        "success": bool(am_pick["success"]),
    }]

    output = []
    for row in rows:
        output.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "repeat_id": int(repeat_id),
            "seed": int(seed),
            "true_arrival": int(metadata["true_arrival"]),
            **row,
        })
    return output

def summarize_metric(raw: pd.DataFrame, metric: str, group_columns: list[str]):
    rows = []
    grouped = raw.groupby(group_columns, sort=True, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[metric].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys))
        row.update({
            "metric": metric,
            "n": int(values.size),
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "total_ms": float(np.sum(values)),
            "total_s": float(np.sum(values) / 1000.0),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("--repeats and --n-jobs must be positive integers.")

    tasks = [
        (frequency, snr_db, repeat_id)
        for frequency in FREQUENCIES
        for snr_db in SNR_VALUES
        for repeat_id in range(args.repeats)
    ]
    seed_rng = np.random.default_rng(BASE_SEED)
    seeds = seed_rng.integers(
        0, np.iinfo(np.uint32).max, size=len(tasks), dtype=np.uint32
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "methods": ["AM-FCM"],
        "frequencies_hz": FREQUENCIES,
        "snr_db": SNR_VALUES,
        "repeats_per_condition": args.repeats,
        "n_jobs": args.n_jobs,
        "base_seed": BASE_SEED,
        "am_c_min": ec.C_MIN,
        "am_c_max": ec.C_MAX,
        "timed_scope_picker": "FCM and arrival decision only",
        "timed_scope_pipeline": "CWT-HOS-iCWT + feature extraction + picker",
        "fcmeans_fallback_used": bool(_baseline_compat.FCM_FALLBACK_USED),
    }
    (args.output_dir / "benchmark_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("AM-FCM cost benchmark")
    print(f"Methods       : {config['methods']}")
    print(f"Frequencies   : {FREQUENCIES} Hz")
    print(f"SNR           : {SNR_VALUES} dB")
    print(f"Repeats       : {args.repeats}/condition")
    print(f"Total traces  : {len(tasks)}")
    print(f"AM clusters   : C={ec.C_MIN}..{ec.C_MAX}")
    print(f"Parallel jobs : {args.n_jobs}")
    print(f"FCM fallback  : {_baseline_compat.FCM_FALLBACK_USED}")
    print("=" * 78)

    warmup_rng = np.random.default_rng(BASE_SEED)
    warmup_noisy, _, _ = ec.simulate(
        am.WAVEFORM, FREQUENCIES[0], SNR_VALUES[0],
        noise_type=am.NOISE_TYPE, noise_bank=None, rng=warmup_rng,
    )
    warmup_enhanced, warmup_kept, _ = ec.cwt_hos_icwt(warmup_noisy)
    warmup_features = (
        None if warmup_kept.size == 0
        else ec.build_feature_matrix(warmup_enhanced, am.FEATURE_NAMES, ec.WINDOW_SIZE)
    )
    ec.adaptive_fcm(warmup_features)

    backend = "loky"
    backend_kwargs = {"n_jobs": args.n_jobs, "inner_max_num_threads": 1}
    start = time.perf_counter()
    with parallel_backend(backend, **backend_kwargs):
        blocks = Parallel(
            n_jobs=args.n_jobs,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(benchmark_one)(frequency, snr_db, repeat_id, int(seed))
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )
    wall_clock_s = time.perf_counter() - start

    raw = pd.DataFrame([row for block in blocks for row in block])
    summary_method = pd.concat([
        summarize_metric(raw, "picker_time_ms", ["method"]),
        summarize_metric(raw, "pipeline_time_ms", ["method"]),
    ], ignore_index=True)
    summary_cluster = pd.concat([
        summarize_metric(raw, metric, ["method", "final_cluster_num"])
        for metric in ["picker_time_ms", "pipeline_time_ms"]
    ], ignore_index=True)
    summary_condition = summarize_metric(
        raw, "picker_time_ms", ["method", "frequency_hz", "snr_db"]
    )
    wall = pd.DataFrame([{
        "n_traces": len(tasks),
        "n_method_records": len(raw),
        "wall_clock_s": wall_clock_s,
        "n_jobs": args.n_jobs,
    }])

    paths = {
        "raw": args.output_dir / "amfcm_algorithm_timing_raw.csv",
        "method": args.output_dir / "amfcm_algorithm_timing_by_method.csv",
        "cluster": args.output_dir / "amfcm_timing_by_final_cluster.csv",
        "condition": args.output_dir / "amfcm_timing_by_frequency_snr.csv",
        "wall": args.output_dir / "amfcm_wallclock.csv",
    }
    raw.to_csv(paths["raw"], index=False, encoding="utf-8-sig")
    summary_method.to_csv(paths["method"], index=False, encoding="utf-8-sig")
    summary_cluster.to_csv(paths["cluster"], index=False, encoding="utf-8-sig")
    summary_condition.to_csv(paths["condition"], index=False, encoding="utf-8-sig")
    wall.to_csv(paths["wall"], index=False, encoding="utf-8-sig")

    print(f"\nAM-FCM benchmark wall-clock: {wall_clock_s:.2f} s")
    print("\nSummary by method:")
    print(summary_method.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAM-FCM summary by final stopping cluster count:")
    print(summary_cluster.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nResult files:")
    for path in paths.values():
        print(path.resolve())

if __name__ == "__main__":
    main()
