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
from sklearn.preprocessing import MinMaxScaler

try:
    from fcmeans import FCM
except ImportError as e:
    from fcm_fallback import FCM

FS = 1000.0
DT = 1.0 / FS
N_SAMPLES = 450
TRUE_PEAK_SAMPLE = 300
WAVELET_LENGTH = 0.10

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6]
REPEATS = 1000

WINDOW_SIZE = 17
NSTA = 20
NLTA = 100

CLUSTER_NUM = 2
FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_ERROR = 1e-6
MEMBERSHIP_THRESHOLD = 0.95

TRUE_ARRIVAL_RATIO = 0.03

BASE_SEED = 20260917
N_JOBS = 6

OUTPUT_DIR = Path(__file__).resolve().parent / "results_old_fcm_multifrequency"

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

def _moving_sum(x: np.ndarray, window_size: int):

    x = np.asarray(x, dtype=float)
    n = x.size
    half = window_size // 2
    idx = np.arange(n)
    starts = np.maximum(0, idx - half)
    ends = np.minimum(n, idx + half + 1)

    c = np.concatenate(([0.0], np.cumsum(x)))
    return c[ends] - c[starts], ends - starts

def get_absolute_amplitude(data: np.ndarray, window_size: int):

    s, length = _moving_sum(np.abs(data), window_size)
    return s / np.maximum(length, 1)

def get_energy(data: np.ndarray, window_size: int):

    s, _ = _moving_sum(np.asarray(data, dtype=float) ** 2, window_size)
    return s

def _causal_mean(data: np.ndarray, period: int):

    x = np.asarray(data, dtype=float)
    n = x.size
    p = int(period)

    idx = np.arange(n)
    starts = np.maximum(0, idx - p + 1)
    ends = idx + 1

    c = np.concatenate(([0.0], np.cumsum(x)))
    length = ends - starts
    return (c[ends] - c[starts]) / np.maximum(length, 1)

def get_SLTA(data: np.ndarray, nsta: int, nlta: int):

    sta = _causal_mean(data, nsta)
    lta = _causal_mean(data, nlta)

    out = np.zeros_like(sta)
    eps = np.finfo(float).eps * max(1.0, float(np.max(np.abs(lta))))
    valid = np.abs(lta) > eps
    out[valid] = sta[valid] / lta[valid]
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def old_fcm_pick(data: np.ndarray):
    abs_amp = get_absolute_amplitude(data, WINDOW_SIZE)
    energy = get_energy(data, WINDOW_SIZE)
    slta = get_SLTA(data, NSTA, NLTA)

    X = np.column_stack([abs_amp, energy, slta])

    scaler = MinMaxScaler()
    X_scale = scaler.fit_transform(X)

    fcm = FCM(
        n_clusters=CLUSTER_NUM,
        max_iter=FCM_MAX_ITER,
        m=FCM_M,
        error=FCM_ERROR,
        random_state=42,
    )
    fcm.fit(X_scale)

    centers = np.asarray(fcm.centers)
    membership = np.asarray(fcm.u)

    strength = centers.mean(axis=1)
    signal_cluster = int(np.argmax(strength))

    signal_membership = membership[:, signal_cluster]
    idx = np.flatnonzero(signal_membership > MEMBERSHIP_THRESHOLD)

    pick = float(idx[0]) if idx.size else np.nan
    return pick

def evaluate_one(frequency: float, snr_db: float, repeat_id: int, seed: int):
    rng = np.random.default_rng(int(seed))

    clean = build_clean_signal(frequency)
    noisy = add_noise(clean, snr_db, rng)

    reference = get_true_arrival(clean)
    pick = old_fcm_pick(noisy)

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
        })

    return pd.DataFrame(rows)

def macro_average_across_frequencies(summary_frequency_snr: pd.DataFrame):

    cols = [
        "success_rate_pct",
        "p_error_le_2ms_pct",
        "p_error_le_5ms_pct",
        "p_error_le_10ms_pct",
        "mae_ms",
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
    print("Old FCM baseline - multi-frequency macro average")
    print("=" * 80)
    print(f"Frequencies       : {FREQUENCIES} Hz")
    print(f"SNR               : {SNR_VALUES} dB")
    print(f"Repeats/condition : {REPEATS}")
    print(f"Features          : Abs amplitude + Energy + STA/LTA")
    print(f"Window/NSTA/NLTA  : {WINDOW_SIZE}/{NSTA}/{NLTA}")
    print(f"FCM               : C={CLUSTER_NUM}, m={FCM_M}")
    print(f"Membership thr.   : {MEMBERSHIP_THRESHOLD}")
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
    raw_path = OUTPUT_DIR / "raw_old_fcm_multifrequency.csv"
    summary_path = OUTPUT_DIR / "summary_old_fcm_by_frequency_snr.csv"
    macro_path = OUTPUT_DIR / "summary_old_fcm_macroavg_by_snr.csv"

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
