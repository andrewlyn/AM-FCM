import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
from scipy.stats import kurtosis
try:
    from joblib import Parallel, delayed, parallel_config
except ImportError:
    from joblib import Parallel, delayed
    from contextlib import nullcontext

    def parallel_config(**kwargs):
        return nullcontext()

RESULT_CSV = Path("./real_amfcm_trace_results.csv")
SIGNAL_DIR = Path("./pre_data/signal")
OUT_DIR = Path("./am_fcm_fig_actual")

FS = 1000.0
DT = 1.0 / FS

WAVELET = "cmor1.5-1.0"
FREQS = np.logspace(np.log10(10), np.log10(500), 100)
WINDOW = 11
SHORT = 21
LONG = 105

FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_TOL = 1e-6
RANDOM_STATE = 42

N_JOBS = 6

def normalize_id(x):
    try:
        return str(int(float(str(x).replace(".txt", ""))))
    except:
        return str(x)

def read_signal(path):
    data = np.loadtxt(path)
    if data.ndim > 1:
        data = data[:, 0]
    return np.asarray(data, dtype=float).ravel()

def moving_mean(x, w):
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")

def sta_lta(x, short=30, long=150, eps=1e-12):
    x = np.asarray(x, dtype=float)
    sta = moving_mean(np.abs(x), short)
    lta = moving_mean(np.abs(x), long)
    return sta / np.maximum(lta, eps)

def minmax_scale(X):
    X = np.asarray(X, dtype=float)
    xmin = np.min(X, axis=0, keepdims=True)
    xmax = np.max(X, axis=0, keepdims=True)
    return (X - xmin) / np.maximum(xmax - xmin, 1e-12)

def window_bounds(n, window_size):
    half = int(window_size) // 2
    idx = np.arange(n)
    starts = np.maximum(0, idx - half)
    ends = np.minimum(n, idx + half + 1)
    return starts, ends

def get_amplitude(data, window_size=WINDOW):
    x = np.abs(np.asarray(data, dtype=float).ravel())
    starts, ends = window_bounds(x.size, window_size)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[ends] - c[starts]) / (ends - starts)

def get_energy(data, window_size=WINDOW):
    x2 = np.asarray(data, dtype=float).ravel() ** 2
    starts, ends = window_bounds(x2.size, window_size)
    c = np.concatenate(([0.0], np.cumsum(x2)))
    return c[ends] - c[starts]

def get_std(data, window_size=WINDOW):
    x = np.asarray(data, dtype=float).ravel()
    starts, ends = window_bounds(x.size, window_size)
    out = np.zeros(x.size, dtype=float)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i] = np.std(x[s:e], ddof=0)
    return out

def get_slta(data, nsta=SHORT, nlta=LONG):
    x2 = np.abs(np.asarray(data, dtype=float).ravel()) ** 2
    n = x2.size
    c = np.concatenate(([0.0], np.cumsum(x2)))
    starts_sta, ends_sta = window_bounds(n, nsta)
    starts_lta, ends_lta = window_bounds(n, nlta)
    sum_sta = c[ends_sta] - c[starts_sta]
    sum_lta = c[ends_lta] - c[starts_lta]
    lens_sta = ends_sta - starts_sta
    lens_lta = ends_lta - starts_lta
    sta = sum_sta / lens_sta
    lta = sum_lta / lens_lta
    return np.divide(sta, lta, out=np.zeros_like(sta), where=lta > 0)

def get_scales_from_freqs(freqs, wavelet, dt):
    fc = pywt.central_frequency(wavelet)
    scales = fc / (freqs * dt)
    return scales

def fuzzy_cmeans(X, c, m=2.0, max_iter=100, tol=1e-6, random_state=42):

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    U = rng.uniform(size=(n, c))
    U = U / np.maximum(U.sum(axis=1, keepdims=True), 1e-12)

    for _ in range(max_iter):
        U_old = U.copy()
        um = U ** m
        centers = (X.T @ um / np.maximum(np.sum(um, axis=0), 1e-12)).T
        dist = np.sqrt(np.einsum("ijk->ij", (X[:, None, :] - centers) ** 2))
        dist = np.maximum(dist, 1e-12)
        temp = dist ** (2.0 / (m - 1.0))
        U = 1.0 / (temp * (1.0 / temp).sum(axis=1, keepdims=True))
        if np.linalg.norm(U - U_old) < tol:
            break

    return centers, U

def extract_amfcm_feature_matrix(signal):

    signal = np.asarray(signal, dtype=float).ravel()

    wavelet = pywt.ContinuousWavelet(WAVELET)

    scales = get_scales_from_freqs(FREQS, WAVELET, DT)
    pad = int(np.ceil(8 * np.max(scales)))
    padded = np.pad(signal, pad, mode="reflect")
    coef, _ = pywt.cwt(
        padded, scales, WAVELET, sampling_period=DT, method="fft"
    )
    coef = coef[:, pad : pad + len(signal)]

    n_scales, n_times = coef.shape
    filtered = coef.copy()
    k = np.zeros(n_scales)
    for i in range(n_scales):
        k[i] = kurtosis(np.abs(coef[i, :]), fisher=True)
    bias = -6.0 / n_times
    variance = 24.0 / n_times
    threshold = np.sqrt(variance / (1.0 - 0.90))
    kept_idx = np.where(np.abs(k - bias) > threshold)[0]
    filtered[np.abs(k - bias) <= threshold, :] = 0

    order = np.argsort(scales)
    scales_sorted = scales[order]
    integrand = np.real(filtered[order, :]) / (scales_sorted[:, None] ** 1.5)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    enhanced = np.asarray(
        trapz(integrand, x=scales_sorted, axis=0), dtype=float
    ).ravel()
    np.nan_to_num(enhanced, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    M = get_amplitude(enhanced, WINDOW)
    Std = get_std(enhanced, WINDOW)

    X = np.column_stack([M, Std])
    X = minmax_scale(X)

    return X, {
        "enhanced": enhanced,
        "kept_idx": kept_idx,
        "feature_names": ["M", "Std"]
    }

def peak_region_bounds(curve, peak, threshold=0.40):
    left = right = int(peak)
    while left > 0 and curve[left - 1] >= threshold:
        left -= 1
    while right + 1 < curve.size and curve[right + 1] >= threshold:
        right += 1
    return left, right

def merge_region_peaks(curve, peaks, prominences):
    peaks = np.asarray(peaks, dtype=int).ravel()
    prominences = np.asarray(prominences, dtype=float).ravel()
    if peaks.size == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(peaks)
    peaks, prominences = peaks[order], prominences[order]
    groups, group, current_right = [], [], -1
    for i, peak in enumerate(peaks):
        left, right = peak_region_bounds(curve, peak)
        if group and left > current_right:
            groups.append(group)
            group = []
        group.append(i)
        current_right = max(current_right, right)
    if group:
        groups.append(group)
    merged = []
    for group in groups:
        best = max(group, key=lambda i: (prominences[i], curve[peaks[i]], -peaks[i]))
        merged.append(int(peaks[best]))
    return np.asarray(merged, dtype=int)

def actual_membership_diagnostic(X, c):

    centers, U = fuzzy_cmeans(
        X, c=c, m=FCM_M, max_iter=FCM_MAX_ITER,
        tol=FCM_TOL, random_state=RANDOM_STATE
    )
    signal_cluster = int(np.argmax(centers.mean(axis=1)))
    signal_membership = U[:, signal_cluster]
    peaks, props = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(
        signal_membership,
        height=0.55,
        prominence=0.12,
        distance=3,
        width=2,
    )
    valid_peaks = merge_region_peaks(
        signal_membership, peaks, props["prominences"]
    )
    return centers, U, signal_cluster, valid_peaks

def plot_one_case(row, signal, centers, U, final_cluster, valid_peaks, out_path):
    t = np.arange(len(signal)) / FS
    final_u = U[:, final_cluster]

    true_idx = row.get("true_index_0", np.nan)
    pred_idx = row.get("predicted_index_0", np.nan)
    peak_idx = row.get("peak_index_0", np.nan)

    status = row.get("status", "")
    stop_reason = row.get("stop_reason", "")
    abs_error = row.get("abs_error_ms", np.nan)
    cluster_num = row.get("cluster_num", np.nan)
    file_id = normalize_id(row.get("file_id", ""))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )
    ax1, ax2 = axes

    ax1.plot(t, signal, lw=0.9, color="black", label="signal")

    if pd.notna(true_idx):
        ax1.axvline(float(true_idx) / FS, color="red", lw=1.2, label="Manual arrival")

    if pd.notna(pred_idx):
        ax1.axvline(float(pred_idx) / FS, color="blue", lw=1.2, ls="--", label="AM-FCM pick")

    if pd.notna(peak_idx):
        ax1.axvline(float(peak_idx) / FS, color="orange", lw=1.0, ls=":", label="Peak index")

    title = (
        f"id={file_id}, status={status}, abs_error_ms={abs_error}, "
        f"C={cluster_num}, signal_cluster={final_cluster + 1}, "
        f"valid_peaks={valid_peaks.tolist()}, stop_reason={stop_reason}"
    )
    ax1.set_title(title, fontsize=10)
    ax1.set_ylabel("Amplitude")
    ax1.grid(ls="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    for k in range(U.shape[1]):
        if k == final_cluster:
            ax2.plot(t, U[:, k], lw=1.6, label=f"final cluster {k + 1}")
        else:
            ax2.plot(t, U[:, k], lw=0.7, color="0.82")

    if pd.notna(true_idx):
        ax2.axvline(float(true_idx) / FS, color="red", lw=1.0)
    if pd.notna(pred_idx):
        ax2.axvline(float(pred_idx) / FS, color="blue", lw=1.0, ls="--")
    if pd.notna(peak_idx):
        ax2.axvline(float(peak_idx) / FS, color="orange", lw=1.0, ls=":")
    for i, peak in enumerate(valid_peaks):
        ax2.axvline(
            float(peak) / FS, color="green", lw=0.9, ls="--",
            label="valid peak" if i == 0 else None
        )

    ax2.set_xlabel("Time / s")
    ax2.set_ylabel("Membership")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(ls="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def process_one(row_dict):
    try:
        file_id = normalize_id(row_dict["file_id"])
        signal_path = SIGNAL_DIR / f"{file_id}.txt"

        if not signal_path.exists():
            return {
                "file_id": file_id,
                "plotted": False,
                "reason": f"missing signal file: {signal_path}"
            }

        signal = read_signal(signal_path)

        c = row_dict.get("cluster_num", np.nan)
        if pd.isna(c):
            return {
                "file_id": file_id,
                "plotted": False,
                "reason": "cluster_num is NaN"
            }

        c = int(round(float(c)))
        if c < 2:
            return {
                "file_id": file_id,
                "plotted": False,
                "reason": f"invalid cluster_num={c}"
            }

        X, extra = extract_amfcm_feature_matrix(signal)
        centers, U, final_cluster, valid_peaks = actual_membership_diagnostic(X, c)

        status = str(row_dict.get("status", ""))
        safe_status = status.replace(" ", "_").replace(">", "gt").replace("/", "_")
        out_name = f"{file_id}_{safe_status}.png"
        out_path = OUT_DIR / out_name

        plot_one_case(
            row=row_dict,
            signal=signal,
            centers=centers,
            U=U,
            final_cluster=final_cluster,
            valid_peaks=valid_peaks,
            out_path=out_path
        )

        return {
            "file_id": file_id,
            "plotted": True,
            "reason": "",
            "cluster_num": c,
            "final_cluster": final_cluster + 1,
            "valid_peaks": valid_peaks.astype(int).tolist(),
            "out_path": str(out_path)
        }

    except Exception as e:
        return {
            "file_id": normalize_id(row_dict.get("file_id", "")),
            "plotted": False,
            "reason": str(e)
        }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")

    mask = (~df["success"].fillna(False)) | (df["abs_error_ms"].fillna(-np.inf) > 50)
    bad_df = df.loc[mask].copy()

    only_ids = os.environ.get("AM_FCM_PLOT_IDS", "").strip()
    if only_ids:
        wanted = {normalize_id(x.strip()) for x in only_ids.split(",") if x.strip()}
        bad_df = bad_df[bad_df["file_id"].map(normalize_id).isin(wanted)].copy()
        print("restricted ids:", sorted(wanted))

    print("total records:", len(df))
    print("selected bad cases:", len(bad_df))
    print(bad_df["status"].value_counts(dropna=False))

    bad_df.to_csv(
        OUT_DIR / "selected_bad_cases.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rows = bad_df.to_dict("records")

    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_one)(row) for row in rows
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUT_DIR / "plot_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plotted_df = result_df[result_df["plotted"] == True]
    skipped_df = result_df[result_df["plotted"] == False]

    plotted_df.to_csv(
        OUT_DIR / "plotted_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    skipped_df.to_csv(
        OUT_DIR / "skipped_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("finished")
    print("plotted:", len(plotted_df))
    print("skipped:", len(skipped_df))
    print("output dir:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
