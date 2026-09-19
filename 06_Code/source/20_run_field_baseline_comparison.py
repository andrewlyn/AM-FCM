from __future__ import annotations
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

DATA_ROOT = Path("./pre_data")
SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"
FS = 1000.0
EXPECTED_SIGNAL_SAMPLES = 1000
LABEL_IS_ONE_BASED = True
ERROR_LIMIT_MS = 50.0
N_JOBS = min(6, max(1, (os.cpu_count() or 2) - 1))

RESULT_FILE = Path("real_baseline_trace_results.csv")
SUMMARY_FILE = Path("real_baseline_summary.csv")
ERROR_OVER_50_FILE = Path("real_baseline_error_over_50ms.csv")
COMPARISON_FILE = Path("real_method_comparison.csv")
AMFCM_RESULT_FILE = Path("real_amfcm_trace_results.csv")

FCM_WINDOW_SIZE = 17
FCM_NSTA = 20
FCM_NLTA = 100
FCM_CLUSTER_NUM = 2
FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_ERROR = 1e-6
FCM_RANDOM_STATE = 42
FCM_MEMBERSHIP_THRESHOLD = 0.95

STALTA_NSTA = 20
STALTA_NLTA = 120
STALTA_THRESHOLD = 3.0

AIC_MIN_SIDE = 2
AIC_EPS = 1e-12

def load_file_ids(signal_dir=SIGNAL_DIR):
    signal_dir = Path(signal_dir)
    if not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory does not exist: {signal_dir}")
    ids = []
    for path in signal_dir.glob("*.txt"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            print(f"Skipping non-numeric file: {path.name}")
    ids = sorted(set(ids))
    if not ids:
        raise FileNotFoundError(f"{signal_dir}  contains no numeric-ID TXT files.")
    return ids

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
        raise ValueError(f"Signal length must be {EXPECTED_SIGNAL_SAMPLES} samples; got {signal.size} samples.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or Inf.")
    label = load_local_label(label_file)
    true_index = label - 1 if LABEL_IS_ONE_BASED else label
    if not 0 <= true_index < signal.size:
        raise ValueError(f"Manual label {label} is outside the signal range.")
    return signal, int(true_index), int(label)

def _moving_sum(x, window_size):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    half = int(window_size) // 2
    idx = np.arange(n)
    starts = np.maximum(0, idx - half)
    ends = np.minimum(n, idx + half + 1)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return c[ends] - c[starts], ends - starts

def fcm_absolute_amplitude(data, window_size=FCM_WINDOW_SIZE):
    s, length = _moving_sum(np.abs(data), window_size)
    return s / np.maximum(length, 1)

def fcm_energy(data, window_size=FCM_WINDOW_SIZE):
    s, _ = _moving_sum(np.asarray(data, dtype=float) ** 2, window_size)
    return s

def _causal_mean_shortened(data, period):
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    p = int(period)
    idx = np.arange(n)
    starts = np.maximum(0, idx - p + 1)
    ends = idx + 1
    c = np.concatenate(([0.0], np.cumsum(x)))
    length = ends - starts
    return (c[ends] - c[starts]) / np.maximum(length, 1)

def fcm_slta(data, nsta=FCM_NSTA, nlta=FCM_NLTA):
    sta = _causal_mean_shortened(data, nsta)
    lta = _causal_mean_shortened(data, nlta)
    out = np.zeros_like(sta)
    eps = np.finfo(float).eps * max(1.0, float(np.max(np.abs(lta))))
    valid = np.abs(lta) > eps
    out[valid] = sta[valid] / lta[valid]
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _init_membership(n_samples, n_clusters, random_state=FCM_RANDOM_STATE):
    rng = np.random.default_rng(int(random_state))
    u = rng.random((n_samples, n_clusters))
    return u / np.sum(u, axis=1, keepdims=True)

def fuzzy_c_means(features, n_clusters=FCM_CLUSTER_NUM, m=FCM_M,
                  max_iter=FCM_MAX_ITER, tol=FCM_ERROR,
                  random_state=FCM_RANDOM_STATE):
    x = np.asarray(features, dtype=float)
    u = _init_membership(x.shape[0], n_clusters, random_state)
    exponent = 2.0 / (m - 1.0)
    tiny = np.finfo(float).tiny
    converged = False
    delta = np.inf
    for iteration in range(1, int(max_iter) + 1):
        um = u ** m
        centers = (um.T @ x) / np.maximum(um.sum(axis=0)[:, None], tiny)
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        u_new = np.zeros_like(u)
        zero_rows = np.any(dist <= 1e-15, axis=1)
        if np.any(zero_rows):
            for i in np.where(zero_rows)[0]:
                zc = np.where(dist[i] <= 1e-15)[0]
                u_new[i, zc] = 1.0 / len(zc)
        nz = ~zero_rows
        if np.any(nz):
            inv = np.maximum(dist[nz], tiny) ** (-exponent)
            u_new[nz] = inv / np.sum(inv, axis=1, keepdims=True)
        delta = float(np.max(np.abs(u_new - u)))
        u = u_new
        if delta < tol:
            converged = True
            break
    um = u ** m
    centers = (um.T @ x) / np.maximum(um.sum(axis=0)[:, None], tiny)
    return centers, u, int(iteration), bool(converged), float(delta)

def pick_chen_fcm(signal):
    features = np.column_stack([
        fcm_absolute_amplitude(signal),
        fcm_energy(signal),
        fcm_slta(signal),
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    x = MinMaxScaler().fit_transform(features)
    centers, membership, iterations, converged, delta = fuzzy_c_means(x)
    signal_cluster = int(np.argmax(centers.mean(axis=1)))
    signal_membership = membership[:, signal_cluster]
    idx = np.flatnonzero(signal_membership > FCM_MEMBERSHIP_THRESHOLD)
    pick = float(idx[0]) if idx.size else np.nan
    return pick, iterations, converged, delta

def _causal_full_mean(x, window):
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

def classic_sta_lta(data, nsta=STALTA_NSTA, nlta=STALTA_NLTA):
    x = np.asarray(data, dtype=float).ravel()
    cf = x ** 2
    sta = _causal_full_mean(cf, nsta)
    lta = _causal_full_mean(cf, nlta)
    ratio = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(sta) & np.isfinite(lta) & (lta > np.finfo(float).tiny)
    ratio[valid] = sta[valid] / lta[valid]
    return ratio

def pick_stalta(signal, threshold=STALTA_THRESHOLD, nsta=STALTA_NSTA, nlta=STALTA_NLTA):
    ratio = classic_sta_lta(signal, nsta, nlta)
    valid_start = nlta - 1
    if valid_start >= ratio.size:
        return np.nan
    r = ratio[valid_start:]
    above = np.zeros_like(r, dtype=bool)
    finite = np.isfinite(r)
    above[finite] = r[finite] >= threshold
    transition = np.flatnonzero((~above[:-1]) & above[1:]) + 1
    if transition.size:
        return float(valid_start + transition[0])
    if above.size and above[0]:
        return float(valid_start)
    return np.nan

def aic_change_point(signal, min_side=AIC_MIN_SIDE):
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
    return float(k[np.argmin(aic)])

def pick_aic(signal):
    try:
        return aic_change_point(signal)
    except Exception:
        return np.nan

def _error_fields(pick, true_index):
    if not np.isfinite(pick):
        return dict(pick=np.nan, signed=np.nan, error=np.nan,
                    le2=False, le5=False, le10=False, success=False)
    pick = int(round(float(pick)))
    signed = (pick - true_index) / FS * 1000.0
    error = abs(signed)
    return dict(pick=pick, signed=float(signed), error=float(error),
                le2=bool(error <= 2.0), le5=bool(error <= 5.0),
                le10=bool(error <= 10.0), success=True)

def run_single_trace(file_id):
    signal, true_index, true_label = load_real_trace(file_id)
    fcm_pick, fcm_iter, fcm_conv, fcm_delta = pick_chen_fcm(signal)
    fcm = _error_fields(fcm_pick, true_index)
    stalta = _error_fields(pick_stalta(signal), true_index)
    aic = _error_fields(pick_aic(signal), true_index)
    return {
        "file_id": int(file_id), "true_index_0": int(true_index), "true_arrival_1": int(true_label),
        "chen_fcm_pick_0": fcm["pick"], "chen_fcm_pick_1": fcm["pick"] + 1 if fcm["success"] else np.nan,
        "chen_fcm_signed_error_ms": fcm["signed"], "chen_fcm_abs_error_ms": fcm["error"],
        "chen_fcm_le2": fcm["le2"], "chen_fcm_le5": fcm["le5"], "chen_fcm_le10": fcm["le10"],
        "chen_fcm_success": fcm["success"], "chen_fcm_iterations": fcm_iter,
        "chen_fcm_converged": fcm_conv, "chen_fcm_delta": fcm_delta,
        "stalta_pick_0": stalta["pick"], "stalta_pick_1": stalta["pick"] + 1 if stalta["success"] else np.nan,
        "stalta_signed_error_ms": stalta["signed"], "stalta_abs_error_ms": stalta["error"],
        "stalta_le2": stalta["le2"], "stalta_le5": stalta["le5"], "stalta_le10": stalta["le10"],
        "stalta_success": stalta["success"],
        "aic_pick_0": aic["pick"], "aic_pick_1": aic["pick"] + 1 if aic["success"] else np.nan,
        "aic_signed_error_ms": aic["signed"], "aic_abs_error_ms": aic["error"],
        "aic_le2": aic["le2"], "aic_le5": aic["le5"], "aic_le10": aic["le10"],
        "aic_success": aic["success"], "status": "OK", "message": "",
    }

def process_one_trace(file_id):
    try:
        return run_single_trace(file_id)
    except Exception as error:
        row = {"file_id": int(file_id), "true_index_0": np.nan, "true_arrival_1": np.nan,
               "status": "Failed", "message": str(error)}
        for prefix in ("chen_fcm", "stalta", "aic"):
            row.update({f"{prefix}_pick_0": np.nan, f"{prefix}_pick_1": np.nan,
                        f"{prefix}_signed_error_ms": np.nan, f"{prefix}_abs_error_ms": np.nan,
                        f"{prefix}_le2": False, f"{prefix}_le5": False, f"{prefix}_le10": False,
                        f"{prefix}_success": False})
        row["chen_fcm_iterations"] = np.nan
        row["chen_fcm_converged"] = False
        row["chen_fcm_delta"] = np.nan
        return row

def summarize_one_method(results, method, prefix):
    total = len(results)
    success = results[f"{prefix}_success"].fillna(False).astype(bool)
    valid = results.loc[success & results[f"{prefix}_abs_error_ms"].notna()].copy()
    return {
        "method": method, "total_records": int(total), "successful_picks": int(len(valid)),
        "success_rate_pct": 100.0 * len(valid) / total if total else np.nan,
        "le2_pct": 100.0 * results[f"{prefix}_le2"].fillna(False).sum() / total if total else np.nan,
        "le5_pct": 100.0 * results[f"{prefix}_le5"].fillna(False).sum() / total if total else np.nan,
        "le10_pct": 100.0 * results[f"{prefix}_le10"].fillna(False).sum() / total if total else np.nan,
        "mae_ms": valid[f"{prefix}_abs_error_ms"].mean() if len(valid) else np.nan,
        "median_ae_ms": valid[f"{prefix}_abs_error_ms"].median() if len(valid) else np.nan,
        "rmse_ms": np.sqrt(np.mean(valid[f"{prefix}_signed_error_ms"] ** 2)) if len(valid) else np.nan,
        "signed_error_ms": valid[f"{prefix}_signed_error_ms"].mean() if len(valid) else np.nan,
        "error_over_50ms": int((valid[f"{prefix}_abs_error_ms"] > ERROR_LIMIT_MS).sum()),
    }

def summarize_results(results):
    return pd.DataFrame([
        summarize_one_method(results, "Chen-FCM", "chen_fcm"),
        summarize_one_method(results, "STA-LTA", "stalta"),
        summarize_one_method(results, "AIC", "aic"),
    ])

def build_error_over_50(results):
    rows = []
    for method, prefix in (("Chen-FCM", "chen_fcm"), ("STA-LTA", "stalta"), ("AIC", "aic")):
        mask = results[f"{prefix}_success"].fillna(False) & (results[f"{prefix}_abs_error_ms"] > ERROR_LIMIT_MS)
        for _, row in results.loc[mask].iterrows():
            rows.append({"method": method, "file_id": int(row["file_id"]), "true_index_0": row["true_index_0"],
                         "pick_index_0": row[f"{prefix}_pick_0"], "signed_error_ms": row[f"{prefix}_signed_error_ms"],
                         "abs_error_ms": row[f"{prefix}_abs_error_ms"]})
    return pd.DataFrame(rows)

def summarize_existing_amfcm(path=AMFCM_RESULT_FILE):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"success", "abs_error_ms", "signed_error_ms"}
    if not required.issubset(df.columns):
        print(f"Detected {path}， but required fields are missing; skipping automatic AM-FCM summary.")
        return None
    total = len(df)
    success = df["success"].fillna(False).astype(bool)
    valid = df.loc[success & df["abs_error_ms"].notna()].copy()
    return {
        "method": "AM-FCM", "total_records": int(total), "successful_picks": int(len(valid)),
        "success_rate_pct": 100.0 * len(valid) / total if total else np.nan,
        "le2_pct": 100.0 * ((df["abs_error_ms"] <= 2) & success).sum() / total if total else np.nan,
        "le5_pct": 100.0 * ((df["abs_error_ms"] <= 5) & success).sum() / total if total else np.nan,
        "le10_pct": 100.0 * ((df["abs_error_ms"] <= 10) & success).sum() / total if total else np.nan,
        "mae_ms": valid["abs_error_ms"].mean() if len(valid) else np.nan,
        "median_ae_ms": valid["abs_error_ms"].median() if len(valid) else np.nan,
        "rmse_ms": np.sqrt(np.mean(valid["signed_error_ms"] ** 2)) if len(valid) else np.nan,
        "signed_error_ms": valid["signed_error_ms"].mean() if len(valid) else np.nan,
        "error_over_50ms": int((valid["abs_error_ms"] > ERROR_LIMIT_MS).sum()),
    }

def run_all_traces(n_jobs=N_JOBS):
    file_ids = load_file_ids()
    print("=" * 76)
    print("Field microseismic baseline: Chen-FCM / STA-LTA / AIC")
    print("=" * 76)
    print(f"Records: {len(file_ids)}")
    print(f"Sampling rate: {FS:g} Hz")
    print(f"Chen-FCM：window={FCM_WINDOW_SIZE}, NSTA/NLTA={FCM_NSTA}/{FCM_NLTA}, C={FCM_CLUSTER_NUM}, membership>{FCM_MEMBERSHIP_THRESHOLD}")
    print(f"STA-LTA：NSTA/NLTA={STALTA_NSTA}/{STALTA_NLTA}, threshold={STALTA_THRESHOLD}")
    print(f"AIC: full trace, global minimum, min_side={AIC_MIN_SIDE}")
    print(f"Parallel workers: {n_jobs}")
    print("<=2/5/10 ms: all records are denominators; missed picks count as misses\n")

    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_one_trace)(file_id) for file_id in file_ids)

    results = pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)
    summary = summarize_results(results)
    over50 = build_error_over_50(results)

    results.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    over50.to_csv(ERROR_OVER_50_FILE, index=False, encoding="utf-8-sig")

    comparison = summary.copy()
    am_row = summarize_existing_amfcm()
    if am_row is not None:
        comparison = pd.concat([pd.DataFrame([am_row]), comparison], ignore_index=True)
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 76)
    print("Final summary")
    print("=" * 76)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    failed = results.loc[results["status"] == "Failed"]
    if len(failed):
        print(f"\nProcessing failures: {len(failed)} records")
        for row in failed.itertuples(index=False):
            print(f"  {int(row.file_id)}.txt：{row.message}")

    print("\nResult files:")
    print(f"1. {RESULT_FILE}")
    print(f"2. {SUMMARY_FILE}")
    print(f"3. {ERROR_OVER_50_FILE}")
    print(f"4. {COMPARISON_FILE}")
    return results, comparison

if __name__ == "__main__":
    run_all_traces()
