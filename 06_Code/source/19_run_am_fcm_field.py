import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.preprocessing import MinMaxScaler

import experiment_common_field as ec

DATA_ROOT = Path("./pre_data")
SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"

RESULT_FILE = Path("real_amfcm_trace_results.csv")
SUMMARY_FILE = Path("real_amfcm_summary.csv")
ERROR_OVER_50_FILE = Path("real_amfcm_error_over_50ms.csv")

FS = float(ec.FS)
DT = 1.0 / FS
EXPECTED_SIGNAL_SAMPLES = 1000
LABEL_IS_ONE_BASED = True
ERROR_LIMIT_MS = 50.0

FEATURE_SET = ("M", "Std")

N_JOBS = min(6, max(1, (os.cpu_count() or 2) - 1))

def load_file_ids(signal_dir=SIGNAL_DIR):

    signal_dir = Path(signal_dir)
    if not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory does not exist: {signal_dir}")

    file_ids = []
    for path in signal_dir.glob("*.txt"):
        try:
            file_ids.append(int(path.stem))
        except ValueError:
            print(f"Skipping non-numeric file: {path.name}")

    file_ids = sorted(set(file_ids))
    if not file_ids:
        raise FileNotFoundError(f"{signal_dir}  contains no numeric-ID TXT files.")
    return file_ids

def load_local_label(label_file):

    values = np.atleast_1d(np.loadtxt(label_file, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Label file contains no valid values.")
    return int(round(float(np.min(values))))

def load_real_trace(file_id):

    signal_file = SIGNAL_DIR / f"{file_id}.txt"
    label_file = LABEL_DIR / f"{file_id}.txt"

    if not signal_file.exists():
        raise FileNotFoundError(f"Missing signal file: {signal_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"Missing label file: {label_file}")

    signal = np.asarray(np.loadtxt(signal_file, dtype=float), dtype=float).ravel()
    if signal.size != EXPECTED_SIGNAL_SAMPLES:
        raise ValueError(
            f"Signal length must be {EXPECTED_SIGNAL_SAMPLES} samples; got {signal.size} samples."
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or Inf.")

    label = load_local_label(label_file)
    true_index = label - 1 if LABEL_IS_ONE_BASED else label

    if not 0 <= true_index < signal.size:
        raise ValueError(
            f"Manual label {label} is outside the signal range 1~{signal.size}."
        )
    return signal, int(true_index), int(label)

def enhance_trace(signal):

    signal = np.asarray(signal, dtype=float).ravel()

    if hasattr(ec, "cwt_hos_icwt"):
        enhanced, kept_idx, _ = ec.cwt_hos_icwt(signal)
        return np.asarray(enhanced, dtype=float).ravel(), np.asarray(kept_idx).ravel()

    coeffs, freqs = ec.cwt_morlet_pywt(signal, ec.DT, ec.CWT_FREQUENCIES)
    filtered, kept_idx = ec.hos_preprocess_cwt(coeffs, freqs)

    if not hasattr(ec, "inverse_cwt"):
        raise AttributeError(
            "experiment_common.py  provides neither cwt_hos_icwt() nor inverse_cwt()."
        )

    enhanced = ec.inverse_cwt(filtered)
    return np.asarray(enhanced, dtype=float).ravel(), np.asarray(kept_idx).ravel()

def build_features(enhanced):

    x = np.asarray(enhanced, dtype=float).ravel()

    feature_bank = {
        "M": np.asarray(ec.get_amplitude(x, ec.WINDOW_SIZE), dtype=float).ravel(),
        "P": np.asarray(ec.get_energy(x, ec.WINDOW_SIZE), dtype=float).ravel(),
        "SLTA": np.asarray(ec.get_SLTA(x, ec.SHORT_SIZE, ec.LONG_SIZE), dtype=float).ravel(),
    }

    if "Std" in FEATURE_SET:
        if not hasattr(ec, "get_std"):
            raise AttributeError("Current experiment_common.py has no get_std().")
        feature_bank["Std"] = np.asarray(
            ec.get_std(x, ec.WINDOW_SIZE), dtype=float
        ).ravel()

    unknown = [name for name in FEATURE_SET if name not in feature_bank]
    if unknown:
        raise ValueError(f"Unsupported feature(s): {unknown}")

    n = min(len(feature_bank[name]) for name in FEATURE_SET)
    features = np.column_stack([feature_bank[name][:n] for name in FEATURE_SET])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if features.shape[0] != x.size:
        raise ValueError(
            f"Feature length {features.shape[0]} differs from record length {x.size}  do not match."
        )

    return MinMaxScaler().fit_transform(features)

def run_single_trace(file_id):
    signal, true_index, true_label = load_real_trace(file_id)

    enhanced, kept_idx = enhance_trace(signal)
    if kept_idx.size == 0:
        raise RuntimeError("HOS retained no valid CWT scales.")

    features = build_features(enhanced)

    result = ec.adaptive_fcm(features)
    if not isinstance(result, dict):
        raise TypeError(
            "The updated experiment_common.adaptive_fcm() should return a dict; "
            f"the current return type is {type(result).__name__}。"
        )

    success = bool(result.get("success", False))
    predicted_index = result.get("arrival", None)

    if (not success) or predicted_index is None:
        return {
            "file_id": int(file_id),
            "true_index_0": int(true_index),
            "true_arrival_1": int(true_label),
            "predicted_index_0": np.nan,
            "predicted_arrival_1": np.nan,
            "signed_error_ms": np.nan,
            "abs_error_ms": np.nan,
            "le2": False,
            "le5": False,
            "le10": False,
            "cluster_num": result.get("cluster_num", np.nan),
            "peak_index_0": result.get("peak", np.nan),
            "stable_count": result.get("stable_count", np.nan),
            "valid_peak_count": result.get("valid_peak_count", np.nan),
            "kept_scale_count": int(kept_idx.size),
            "success": False,
            "status": "Unreliable",
            "stop_reason": result.get("stop_reason", ""),
            "message": result.get("diagnostic_message", ""),
        }

    predicted_index = int(predicted_index)
    if not 0 <= predicted_index < signal.size:
        raise ValueError(f"Predicted arrival {predicted_index} is outside the waveform range.")

    signed_error_ms = (predicted_index - true_index) / FS * 1000.0
    abs_error_ms = abs(signed_error_ms)

    return {
        "file_id": int(file_id),
        "true_index_0": int(true_index),
        "true_arrival_1": int(true_label),
        "predicted_index_0": int(predicted_index),
        "predicted_arrival_1": int(predicted_index + 1),
        "signed_error_ms": float(signed_error_ms),
        "abs_error_ms": float(abs_error_ms),
        "le2": bool(abs_error_ms <= 2.0),
        "le5": bool(abs_error_ms <= 5.0),
        "le10": bool(abs_error_ms <= 10.0),
        "cluster_num": result.get("cluster_num", np.nan),
        "peak_index_0": result.get("peak", np.nan),
        "stable_count": result.get("stable_count", np.nan),
        "valid_peak_count": result.get("valid_peak_count", np.nan),
        "kept_scale_count": int(kept_idx.size),
        "success": True,
        "status": "Reliable",
        "stop_reason": result.get("stop_reason", ""),
        "message": result.get("diagnostic_message", ""),
    }

def process_one_trace(file_id):

    try:
        row = run_single_trace(file_id)
        if row["success"] and row["abs_error_ms"] > ERROR_LIMIT_MS:
            row["status"] = f"Error > {ERROR_LIMIT_MS:g} ms"
        return row

    except Exception as error:
        return {
            "file_id": int(file_id),
            "true_index_0": np.nan,
            "true_arrival_1": np.nan,
            "predicted_index_0": np.nan,
            "predicted_arrival_1": np.nan,
            "signed_error_ms": np.nan,
            "abs_error_ms": np.nan,
            "le2": False,
            "le5": False,
            "le10": False,
            "cluster_num": np.nan,
            "peak_index_0": np.nan,
            "stable_count": np.nan,
            "valid_peak_count": np.nan,
            "kept_scale_count": np.nan,
            "success": False,
            "status": "Failed",
            "stop_reason": "",
            "message": str(error),
        }

def summarize_results(results):

    total = len(results)
    success_mask = results["success"].fillna(False).astype(bool)
    reliable = results.loc[
        success_mask & results["abs_error_ms"].notna()
    ].copy()

    failed = results.loc[results["status"] == "Failed"]
    unreliable = results.loc[results["status"] == "Unreliable"]
    over50 = reliable.loc[reliable["abs_error_ms"] > ERROR_LIMIT_MS]

    summary = {
        "feature_set": "+".join(FEATURE_SET),
        "total_records": int(total),
        "reliable_picks": int(len(reliable)),
        "reliable_rate_pct": 100.0 * len(reliable) / total if total else np.nan,
        "unreliable_picks": int(len(unreliable)),
        "failed_records": int(len(failed)),
        "le2_pct": 100.0 * results["le2"].fillna(False).sum() / total if total else np.nan,
        "le5_pct": 100.0 * results["le5"].fillna(False).sum() / total if total else np.nan,
        "le10_pct": 100.0 * results["le10"].fillna(False).sum() / total if total else np.nan,
        "mae_ms": reliable["abs_error_ms"].mean() if len(reliable) else np.nan,
        "median_ae_ms": reliable["abs_error_ms"].median() if len(reliable) else np.nan,
        "rmse_ms": (
            np.sqrt(np.mean(reliable["signed_error_ms"] ** 2))
            if len(reliable) else np.nan
        ),
        "signed_error_ms": reliable["signed_error_ms"].mean() if len(reliable) else np.nan,
        "error_over_50ms": int(len(over50)),
        "error_over_50ms_pct_all": 100.0 * len(over50) / total if total else np.nan,
    }
    return pd.DataFrame([summary]), reliable, unreliable, failed, over50

def run_all_traces(n_jobs=N_JOBS):
    file_ids = load_file_ids()

    print("=" * 70)
    print("AM-FCM picking on field microseismic records")
    print("=" * 70)
    print(f"Records: {len(file_ids)}")
    print(f"Sampling rate: {FS:g} Hz")
    print(f"Input features: {'+'.join(FEATURE_SET)}")
    print(f"Feature window: {ec.WINDOW_SIZE}")
    print(f"AM-FCM：experiment_common.adaptive_fcm")
    print(f"Parallel workers: {n_jobs}")
    print("≤2/5/10 ms: all records are denominators; unreliable picks count as failures")
    print()

    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_one_trace)(file_id) for file_id in file_ids
        )

    results = pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)
    summary, reliable, unreliable, failed, over50 = summarize_results(results)

    results.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    over50.to_csv(ERROR_OVER_50_FILE, index=False, encoding="utf-8-sig")

    s = summary.iloc[0]

    print("\n" + "=" * 70)
    print("Final summary")
    print("=" * 70)
    print(f"Total records: {int(s.total_records)}")
    print(f"Reliable picks: {int(s.reliable_picks)} ({s.reliable_rate_pct:.2f}%)")
    print(f"Unreliable picks: {int(s.unreliable_picks)}")
    print(f"Processing failures: {int(s.failed_records)}")
    print(f"≤2 ms： {s.le2_pct:.2f}%")
    print(f"≤5 ms： {s.le5_pct:.2f}%")
    print(f"≤10 ms：{s.le10_pct:.2f}%")
    print(f"MAE：{s.mae_ms:.3f} ms")
    print(f"Median AE：{s.median_ae_ms:.3f} ms")
    print(f"RMSE：{s.rmse_ms:.3f} ms")
    print(f"Signed error：{s.signed_error_ms:+.3f} ms")
    print(f">50 ms: {int(s.error_over_50ms)} records")

    if len(unreliable):
        print(f"\nUnreliable record IDs: {unreliable['file_id'].astype(int).tolist()}")
    if len(failed):
        print(f"\nFailed record IDs: {failed['file_id'].astype(int).tolist()}")
        for row in failed.itertuples(index=False):
            print(f"  {int(row.file_id)}.txt：{row.message}")
    if len(over50):
        print(f"\nRecord IDs with >50 ms error: {over50['file_id'].astype(int).tolist()}")

    print("\nResult files:")
    print(f"1. {RESULT_FILE}")
    print(f"2. {SUMMARY_FILE}")
    print(f"3. {ERROR_OVER_50_FILE}")

    return results, summary

if __name__ == "__main__":
    run_all_traces()
