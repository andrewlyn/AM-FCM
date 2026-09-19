from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from experiment_common import DT, load_real_noise_csv, simulate

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6, 2]
REPEATS = 200

AIC_MIN_SIDE = 2
AIC_EPS = 1e-12
ERROR_LIMIT_MS = 100.0
BASE_SEED = 20260911

NOISE_TYPE = "WGN"
N_JOBS = 6
OUTPUT_DIR = Path(__file__).resolve().parent / "results_aic"

def _first_finite_pick(pick: object) -> float:

    if pick is None:
        return np.nan
    try:
        value = float(pick)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan

def _aic_change_point(signal: np.ndarray, min_side: int = AIC_MIN_SIDE) -> int:

    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n < 2 * min_side + 1:
        raise ValueError("Signal is too short to compute AIC.")

    k = np.arange(min_side, n - min_side, dtype=int)
    cs = np.cumsum(x)
    cq = np.cumsum(x ** 2)

    n1 = k.astype(float)
    s1 = cs[k - 1]
    q1 = cq[k - 1]
    var1 = q1 / n1 - (s1 / n1) ** 2

    n2 = (n - k).astype(float)
    s2 = cs[-1] - s1
    q2 = cq[-1] - q1
    var2 = q2 / n2 - (s2 / n2) ** 2

    var1 = np.maximum(var1, AIC_EPS)
    var2 = np.maximum(var2, AIC_EPS)
    aic = n1 * np.log(var1) + n2 * np.log(var2)
    return int(k[np.argmin(aic)])

def pick_clean_aic(clean: np.ndarray) -> float:

    try:
        x = np.asarray(clean, dtype=float).ravel()
        peak_index = int(np.argmax(np.abs(x)))
        return _first_finite_pick(_aic_change_point(x[: peak_index + 1]))
    except Exception:
        return np.nan

def pick_aic_noisy(noisy: np.ndarray) -> float:

    try:
        return _first_finite_pick(_aic_change_point(np.asarray(noisy, dtype=float).ravel()))
    except Exception:
        return np.nan

def evaluate_one(frequency: float, snr_db: float, repeat_id: int, seed: int, noise_bank):

    rng = np.random.default_rng(int(seed))
    noise_type = NOISE_TYPE if noise_bank is None else "real"

    np.random.seed(int(seed))
    noisy, clean, metadata = simulate(
        waveform="ricker",
        frequency=frequency,
        snr_db=snr_db,
        noise_type=noise_type,
        noise_bank=noise_bank,
        rng=rng,
    )
    noisy = np.asarray(noisy, dtype=float).ravel()
    clean = np.asarray(clean, dtype=float).ravel()

    reference_arrival = pick_clean_aic(clean)
    if not np.isfinite(reference_arrival):
        raise RuntimeError("Failed to compute the AIC reference arrival for the clean waveform.")
    reference_arrival = int(reference_arrival)

    common_reference = int(metadata["true_arrival"])
    if reference_arrival != common_reference:
        raise RuntimeError(
            f"Independent clean-AIC arrival({reference_arrival}) and the value returned by simulate"
            f"({common_reference}) do not match."
        )

    aic_pick = pick_aic_noisy(noisy)
    signed_error_samples = aic_pick - reference_arrival if np.isfinite(aic_pick) else np.nan
    error_samples = abs(signed_error_samples) if np.isfinite(signed_error_samples) else np.nan
    error_ms = error_samples * DT * 1000.0 if np.isfinite(error_samples) else np.nan
    signed_error_ms = signed_error_samples * DT * 1000.0 if np.isfinite(signed_error_samples) else np.nan

    return {
        "frequency_hz": int(frequency),
        "snr_db": float(snr_db),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "reference_arrival_sample": int(reference_arrival),
        "reference_arrival_ms": float(reference_arrival * DT * 1000.0),
        "common_reference_arrival_sample": int(common_reference),
        "aic_pick_sample": aic_pick,
        "aic_signed_error_samples": signed_error_samples,
        "aic_error_samples": error_samples,
        "aic_signed_error_ms": signed_error_ms,
        "aic_error_ms": error_ms,
        "aic_success": bool(np.isfinite(aic_pick)),
        "noise_type": noise_type,
        "noise_source_id": int(metadata["noise_source_id"]),
        "noise_start_sample": int(metadata["noise_start_sample"]),
    }

PROBABILITY_LIMITS_MS = (2.0, 5.0, 10.0)

def summarize_probability(raw: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:

    rows = []
    grouped = raw.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), raw)]

    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        error = group["aic_error_ms"].to_numpy(dtype=float)
        signed = group["aic_signed_error_ms"].to_numpy(dtype=float)
        finite = np.isfinite(error)

        row = dict(zip(group_columns, keys)) if group_columns else {}
        row["n"] = int(len(group))
        row["n_success"] = int(finite.sum())
        row["success_rate_pct"] = float(100.0 * finite.mean())
        row["mae_ms"] = float(np.nanmean(error)) if finite.any() else np.nan
        row["mean_signed_error_ms"] = float(np.nanmean(signed)) if finite.any() else np.nan

        for limit_ms in PROBABILITY_LIMITS_MS:
            label = str(int(limit_ms))
            row[f"p_error_le_{label}ms_pct"] = float(100.0 * (finite & (error <= limit_ms)).mean())
        rows.append(row)

    return pd.DataFrame(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--frequencies", nargs="+", type=int, default=FREQUENCIES)
    parser.add_argument("--snrs", nargs="+", type=float, default=SNR_VALUES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--noise-csv", type=Path, default=None, help="Optional real-noise CSV; WGN is used when omitted.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer.")

    noise_bank = load_real_noise_csv(args.noise_csv) if args.noise_csv else None
    tasks = [
        (frequency, snr_db, repeat_id)
        for frequency in args.frequencies
        for snr_db in args.snrs
        for repeat_id in range(args.repeats)
    ]
    seed_rng = np.random.default_rng(args.base_seed)
    seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=len(tasks), dtype=np.uint32)

    print("=" * 72)
    print("Independent AIC arrival-picking experiment")
    print(f"Frequencies: {args.frequencies} Hz")
    print(f"SNR: {args.snrs} dB")
    print(f"Repeats per condition: {args.repeats}")
    print(f"Total records: {len(tasks)}")
    print(f"Noise: {'real' if noise_bank is not None else 'WGN'}")
    print(f"Parallel jobs: {args.n_jobs}")
    print("=" * 72)

    with parallel_backend("loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=args.n_jobs,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(frequency, snr_db, repeat_id, seed, noise_bank)
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )

    raw = pd.DataFrame(results)
    summary_condition = summarize_probability(raw, ["frequency_hz", "snr_db"])
    summary_snr = summarize_probability(raw, ["snr_db"])
    summary_overall = summarize_probability(raw, [])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw_aic_results.csv", index=False, encoding="utf-8-sig")
    summary_condition.to_csv(
        args.output_dir / "summary_aic_by_frequency_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_snr.to_csv(
        args.output_dir / "summary_aic_by_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_overall.to_csv(
        args.output_dir / "summary_aic_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAIC summary by frequency and SNR")
    print(summary_condition.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nAIC summary by SNR")
    print(summary_snr.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nAIC overall summary")
    print(summary_overall.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
