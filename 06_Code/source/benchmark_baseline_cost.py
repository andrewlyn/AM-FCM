from __future__ import annotations

import argparse
import json
import sys
import types
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

def _install_fcmeans_fallback_if_needed():

    try:
        import fcmeans  # noqa: F401
        return False
    except ImportError:
        pass

    class FCM:
        def __init__(self, n_clusters=5, max_iter=150, m=2.0,
                     error=1e-5, random_state=None, **kwargs):
            self.n_clusters = int(n_clusters)
            self.max_iter = int(max_iter)
            self.m = float(m)
            self.error = float(error)
            self.random_state = random_state
            self.trained = False

        @staticmethod
        def _dist(A, B):
            return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            rng = np.random.default_rng(self.random_state)
            n_samples = X.shape[0]
            self.u = rng.uniform(size=(n_samples, self.n_clusters))
            self.u /= self.u.sum(axis=1, keepdims=True)
            for _ in range(self.max_iter):
                u_old = self.u.copy()
                um = self.u ** self.m
                self._centers = (X.T @ um / np.sum(um, axis=0)).T
                temp = self._dist(X, self._centers) ** (2 / (self.m - 1))
                self.u = 1.0 / (temp * (1.0 / temp).sum(axis=1, keepdims=True))
                if np.linalg.norm(self.u - u_old) < self.error:
                    break
            self.trained = True

        @property
        def centers(self):
            if not self.trained:
                raise ReferenceError("FCM has not been fitted.")
            return self._centers

    module = types.ModuleType("fcmeans")
    module.FCM = FCM
    sys.modules["fcmeans"] = module
    return True

FCM_FALLBACK_USED = _install_fcmeans_fallback_if_needed()

import chen_fcm_benchmark as chen
import stalta_benchmark as stalta
import aic_benchmark as aic

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6]
REPEATS = 1000
N_JOBS = 6
BASE_SEED = 20260917
OUTPUT_DIR = Path(__file__).resolve().parent / "results_baseline_cost"

def _simulate_for_method(method: str, frequency: int, snr_db: float, seed: int):

    rng = np.random.default_rng(int(seed))

    if method == "AIC":
        np.random.seed(int(seed))
        noisy, _, _ = aic.simulate(
            waveform="ricker",
            frequency=frequency,
            snr_db=snr_db,
            noise_type="WGN",
            noise_bank=None,
            rng=rng,
        )
        return np.asarray(noisy, dtype=float).ravel()

    if method == "STA/LTA":
        clean = stalta.build_clean_signal(frequency)
        return stalta.add_noise(clean, snr_db, rng)

    if method == "Chen-FCM":
        clean = chen.build_clean_signal(frequency)
        return chen.add_noise(clean, snr_db, rng)

    raise ValueError(f"Unknown method: {method}")

def _pick(method: str, noisy: np.ndarray):
    if method == "AIC":
        return aic.pick_aic_noisy(noisy)
    if method == "STA/LTA":
        return stalta.stalta_pick(noisy)
    if method == "Chen-FCM":
        return chen.old_fcm_pick(noisy)
    raise ValueError(f"Unknown method: {method}")

def benchmark_one(method: str, frequency: int, snr_db: float,
                  repeat_id: int, seed: int):
    noisy = _simulate_for_method(method, frequency, snr_db, seed)

    start_ns = time.perf_counter_ns()
    pick = _pick(method, noisy)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    return {
        "method": method,
        "frequency_hz": int(frequency),
        "snr_db": float(snr_db),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "algorithm_time_ms": float(elapsed_ms),
        "pick_sample": float(pick) if pick is not None and np.isfinite(pick) else np.nan,
    }

def summarize_timing(raw: pd.DataFrame, group_columns: list[str]):
    rows = []
    grouped = raw.groupby(group_columns, sort=True) if group_columns else [((), raw)]

    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        values = group["algorithm_time_ms"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys)) if group_columns else {}
        row.update({
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
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer.")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be a positive integer.")

    tasks = [
        (method, frequency, snr_db, repeat_id)
        for method in ["AIC", "STA/LTA", "Chen-FCM"]
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
        "methods": ["AIC", "STA/LTA", "Chen-FCM"],
        "frequencies_hz": FREQUENCIES,
        "snr_db": SNR_VALUES,
        "repeats_per_condition": args.repeats,
        "n_jobs": args.n_jobs,
        "base_seed": BASE_SEED,
        "timed_scope": "picker only; excludes simulation/reference/I/O/parallel scheduling",
        "fcmeans_fallback_used": bool(FCM_FALLBACK_USED),
    }
    (args.output_dir / "benchmark_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("Baseline algorithm cost benchmark")
    print(f"Methods       : {config['methods']}")
    print(f"Frequencies   : {FREQUENCIES} Hz")
    print(f"SNR           : {SNR_VALUES} dB")
    print(f"Repeats       : {args.repeats}/condition")
    print(f"Total records : {len(tasks)}")
    print(f"Parallel jobs : {args.n_jobs}")
    print("Timed scope   : picker only")
    print(f"FCM fallback  : {FCM_FALLBACK_USED}")
    print("=" * 78)

    for method in ["AIC", "STA/LTA", "Chen-FCM"]:
        warmup_seed = int(seed_rng.integers(0, np.iinfo(np.uint32).max))
        noisy = _simulate_for_method(method, FREQUENCIES[0], SNR_VALUES[0], warmup_seed)
        _pick(method, noisy)

    wall_rows = []
    all_results = []
    for method in ["AIC", "STA/LTA", "Chen-FCM"]:
        method_tasks = [
            (frequency, snr_db, repeat_id, int(seed))
            for (task_method, frequency, snr_db, repeat_id), seed in zip(tasks, seeds)
            if task_method == method
        ]
        start = time.perf_counter()
        backend = "loky"
        backend_kwargs = {"n_jobs": args.n_jobs, "inner_max_num_threads": 1}
        with parallel_backend(backend, **backend_kwargs):
            method_results = Parallel(
                n_jobs=args.n_jobs,
                batch_size="auto",
                pre_dispatch="2*n_jobs",
                verbose=10,
            )(
                delayed(benchmark_one)(method, frequency, snr_db, repeat_id, seed)
                for frequency, snr_db, repeat_id, seed in method_tasks
            )
        wall_seconds = time.perf_counter() - start
        all_results.extend(method_results)
        timing_values = np.asarray(
            [row["algorithm_time_ms"] for row in method_results], dtype=float
        )
        wall_rows.append({
            "method": method,
            "n_records": int(len(method_results)),
            "algorithm_total_s": float(timing_values.sum() / 1000.0),
            "wall_clock_s": float(wall_seconds),
            "n_jobs": int(args.n_jobs),
        })
        print(f"{method}: wall-clock {wall_seconds:.2f} s; "
              f"picker total {timing_values.sum() / 1000.0:.2f} s")

    raw = pd.DataFrame(all_results)
    summary_condition = summarize_timing(raw, ["method", "frequency_hz", "snr_db"])
    summary_method = summarize_timing(raw, ["method"])
    wall = pd.DataFrame(wall_rows)

    raw_path = args.output_dir / "baseline_algorithm_timing_raw.csv"
    condition_path = args.output_dir / "baseline_algorithm_timing_by_frequency_snr.csv"
    method_path = args.output_dir / "baseline_algorithm_timing_by_method.csv"
    wall_path = args.output_dir / "baseline_algorithm_wallclock.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_condition.to_csv(condition_path, index=False, encoding="utf-8-sig")
    summary_method.to_csv(method_path, index=False, encoding="utf-8-sig")
    wall.to_csv(wall_path, index=False, encoding="utf-8-sig")

    print("\nSummary by algorithm (picker cost):")
    print(summary_method.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nResult files:")
    for path in [raw_path, condition_path, method_path, wall_path]:
        print(path.resolve())

if __name__ == "__main__":
    main()
