from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

FS = 1000.0
DT = 1.0 / FS
N_SAMPLES = 450
TRUE_PEAK_SAMPLE = 300
WAVELET_LENGTH = 0.10

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6]
REPEATS = 1000

NSTA = 20
NLTA = 120
TRIGGER_THRESHOLD = 3.0

TRUE_ARRIVAL_RATIO = 0.03

BASE_SEED = 20260917
N_JOBS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "results_stalta_multifrequency"

def ricker_wavelet(frequency: float, fs: float = FS, length: float = WAVELET_LENGTH):

    dt = 1.0 / fs
    t = np.arange(-length / 2.0, length / 2.0 + 0.5 * dt, dt)
    a = (np.pi * float(frequency) * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(float)

def embed_wavelet(wavelet: np.ndarray, n_samples: int = N_SAMPLES,
                  peak_sample: int = TRUE_PEAK_SAMPLE):

    w = np.asarray(wavelet, dtype=float).ravel()
    x = np.zeros(int(n_samples), dtype=float)

    local_peak = int(np.argmax(np.abs(w)))
    start = int(peak_sample) - local_peak
    src_start = max(0, -start)
    dst_start = max(0, start)
    copy_len = min(w.size - src_start, x.size - dst_start)

    if copy_len > 0:
        x[dst_start:dst_start + copy_len] = w[src_start:src_start + copy_len]
    return x

def add_noise(clean: np.ndarray, snr_db: float, rng: np.random.Generator):

    x = np.asarray(clean, dtype=float)
    signal_power = float(np.mean(x ** 2))

    noise = rng.standard_normal(x.size)
    noise_power = float(np.mean(noise ** 2))

    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(target_noise_power / max(noise_power, np.finfo(float).tiny))
    return x + scale * noise

def build_clean_signal(frequency: float):
    return embed_wavelet(ricker_wavelet(frequency))

def get_true_arrival(clean: np.ndarray):

    threshold = TRUE_ARRIVAL_RATIO * np.max(np.abs(clean))
    idx = np.flatnonzero(np.abs(clean) > threshold)
    return int(idx[0]) if idx.size else np.nan

def causal_full_mean(x: np.ndarray, window: int):

    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    w = int(window)

    out = np.full(n, np.nan, dtype=float)
    if w <= 0 or w > n:
        return out

    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(w - 1, n)
    ends = idx + 1
    starts = ends - w
    out[idx] = (c[ends] - c[starts]) / float(w)
    return out

def classic_sta_lta(data: np.ndarray, nsta: int = NSTA, nlta: int = NLTA):

    if nlta <= nsta:
        raise ValueError("NLTA must be greater than NSTA.")

    x = np.asarray(data, dtype=float).ravel()
    cf = x ** 2

    sta = causal_full_mean(cf, nsta)
    lta = causal_full_mean(cf, nlta)

    ratio = np.full_like(x, np.nan, dtype=float)

    valid = (
        np.isfinite(sta)
        & np.isfinite(lta)
        & (lta > np.finfo(float).tiny)
    )
    ratio[valid] = sta[valid] / lta[valid]
    return ratio

def stalta_pick(data: np.ndarray, threshold: float = TRIGGER_THRESHOLD,
                nsta: int = NSTA, nlta: int = NLTA):

    ratio = classic_sta_lta(data, nsta, nlta)

    valid_start = nlta - 1
    if valid_start >= len(ratio):
        return np.nan

    r = ratio[valid_start:]
    finite = np.isfinite(r)

    above = np.zeros_like(r, dtype=bool)
    above[finite] = r[finite] >= threshold

    transition = np.flatnonzero((~above[:-1]) & above[1:]) + 1

    if transition.size:
        return float(valid_start + transition[0])

    if above.size and above[0]:
        return float(valid_start)

    return np.nan

def evaluate_one(frequency: float, snr_db: float, repeat_id: int, seed: int):
    rng = np.random.default_rng(int(seed))

    clean = build_clean_signal(frequency)
    noisy = add_noise(clean, snr_db, rng)

    reference = get_true_arrival(clean)
    pick = stalta_pick(noisy)

    success = np.isfinite(reference) and np.isfinite(pick)

    if success:
        signed_error_ms = (pick - reference) * DT * 1000.0
        error_ms = abs(signed_error_ms)
    else:
        signed_error_ms = np.nan
        error_ms = np.nan

    return {
        "frequency_hz": int(frequency),
        "snr_db": float(snr_db),
        "repeat_id": int(repeat_id),
        "reference_sample": reference,
        "pick_sample": pick,
        "success": bool(success),
        "signed_error_ms": signed_error_ms,
        "error_ms": error_ms,
    }

def summarize_frequency_snr(raw: pd.DataFrame):
    rows = []

    for (frequency, snr_db), g in raw.groupby(["frequency_hz", "snr_db"], sort=True):
        err = g["error_ms"].to_numpy(dtype=float)
        signed = g["signed_error_ms"].to_numpy(dtype=float)
        finite = np.isfinite(err)

        rows.append({
            "frequency_hz": int(frequency),
            "snr_db": float(snr_db),
            "n": int(len(g)),
            "n_success": int(finite.sum()),
            "success_rate_pct": 100.0 * float(finite.mean()),
            "p_error_le_2ms_pct": 100.0 * float((finite & (err <= 2.0)).mean()),
            "p_error_le_5ms_pct": 100.0 * float((finite & (err <= 5.0)).mean()),
            "p_error_le_10ms_pct": 100.0 * float((finite & (err <= 10.0)).mean()),
            "mae_ms": float(np.nanmean(err)) if finite.any() else np.nan,
            "median_ae_ms": float(np.nanmedian(err)) if finite.any() else np.nan,
            "mean_signed_error_ms": float(np.nanmean(signed)) if finite.any() else np.nan,
        })

    return pd.DataFrame(rows)

def macro_average_across_frequencies(summary_frequency_snr: pd.DataFrame):

    cols = [
        "success_rate_pct",
        "p_error_le_2ms_pct",
        "p_error_le_5ms_pct",
        "p_error_le_10ms_pct",
        "mae_ms",
        "median_ae_ms",
        "mean_signed_error_ms",
    ]

    return (
        summary_frequency_snr
        .groupby("snr_db", as_index=False)[cols]
        .mean()
        .sort_values("snr_db")
        .reset_index(drop=True)
    )

def main():
    tasks = [
        (frequency, snr_db, repeat_id)
        for frequency in FREQUENCIES
        for snr_db in SNR_VALUES
        for repeat_id in range(REPEATS)
    ]

    seed_rng = np.random.default_rng(BASE_SEED)
    seeds = seed_rng.integers(
        0, np.iinfo(np.uint32).max,
        size=len(tasks), dtype=np.uint32
    )

    print("=" * 80)
    print("STA/LTA baseline - multi-frequency macro average")
    print("=" * 80)
    print(f"Frequencies       : {FREQUENCIES} Hz")
    print(f"SNR               : {SNR_VALUES} dB")
    print(f"Repeats/condition : {REPEATS}")
    print(f"NSTA/NLTA         : {NSTA}/{NLTA}")
    print(f"Trigger threshold : {TRIGGER_THRESHOLD}")
    print("Characteristic    : energy x^2")
    print(f"Reference         : clean |x| > {TRUE_ARRIVAL_RATIO:.2f}*peak")
    print(f"Total records     : {len(tasks)}")
    print("=" * 80)

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=N_JOBS,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(
                frequency, snr_db, repeat_id, int(seed)
            )
            for (frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
        )

    raw = pd.DataFrame(results)
    summary = summarize_frequency_snr(raw)
    macro = macro_average_across_frequencies(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / "raw_stalta_multifrequency.csv"
    summary_path = OUTPUT_DIR / "summary_stalta_by_frequency_snr.csv"
    macro_path = OUTPUT_DIR / "summary_stalta_macroavg_by_snr.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")

    print("\nBy frequency × SNR:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nFinal results: equal-weight average across dominant frequencies")
    print(macro.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved files:")
    print(raw_path.resolve())
    print(summary_path.resolve())
    print(macro_path.resolve())

if __name__ == "__main__":
    main()
