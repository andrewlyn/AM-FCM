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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8

plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

plt.rcParams["axes.unicode_minus"] = False
from sklearn.preprocessing import MinMaxScaler

import experiment_common_field as ec

DATA_ROOT = Path("./pre_data")
SIGNAL_DIR = DATA_ROOT / "signal"
RESULT_CSV = Path("./real_amfcm_trace_results.csv")

OUTPUT_TIFF = Path("./am_fcm_representative_cases.tiff")

SUCCESS_ID = 267
UNRELIABLE_ID = 78

FS = float(ec.FS)
EXPECTED_SAMPLES = 1000

FEATURE_SET = ("M", "Std")

FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_TOL = 1e-6
RANDOM_STATE = 42

def read_signal(file_id):

    path = SIGNAL_DIR / f"{file_id}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Missing signal file: {path}")

    signal = np.asarray(np.loadtxt(path, dtype=float), dtype=float).ravel()

    if signal.size != EXPECTED_SAMPLES:
        raise ValueError(
            f"{path.name}: signal length={signal.size}，"
            f"expected={EXPECTED_SAMPLES}"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{path.name}: signal contains NaN or Inf")

    return signal

def load_result_table():

    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"Result files not found: {RESULT_CSV}")

    df = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")

    required = {"file_id", "true_index_0", "cluster_num", "peak_index_0"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Result file is missing fields: {sorted(missing)}")

    df["file_id"] = df["file_id"].astype(int)
    return df

def get_result_row(df, file_id):
    rows = df[df["file_id"] == file_id]

    if len(rows) == 0:
        raise ValueError(f"Record {file_id} not found in the result file")

    return rows.iloc[0]

def enhance_trace(signal):

    signal = np.asarray(signal, dtype=float).ravel()

    if hasattr(ec, "cwt_hos_icwt"):
        enhanced, kept_idx, _ = ec.cwt_hos_icwt(signal)
        return np.asarray(enhanced, dtype=float).ravel(), np.asarray(kept_idx).ravel()

    coeffs, freqs = ec.cwt_morlet_pywt(signal, ec.DT, ec.CWT_FREQUENCIES)
    filtered, kept_idx = ec.hos_preprocess_cwt(coeffs, freqs)

    if not hasattr(ec, "inverse_cwt"):
        raise AttributeError(
            "experiment_common.py provides neither cwt_hos_icwt() nor inverse_cwt()."
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
            raise AttributeError("experiment_common.py has no get_std().")
        feature_bank["Std"] = np.asarray(
            ec.get_std(x, ec.WINDOW_SIZE), dtype=float
        ).ravel()

    unknown = [name for name in FEATURE_SET if name not in feature_bank]
    if unknown:
        raise ValueError(f"Unsupported feature(s): {unknown}")

    features = np.column_stack([feature_bank[name] for name in FEATURE_SET])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if features.shape[0] != EXPECTED_SAMPLES:
        raise ValueError(
            f"feature length={features.shape[0]}, "
            f"signal length={EXPECTED_SAMPLES}"
        )

    return MinMaxScaler().fit_transform(features)

def fuzzy_cmeans(X, c, m=FCM_M, max_iter=FCM_MAX_ITER,
                 tol=FCM_TOL, random_state=RANDOM_STATE):

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)

    n = X.shape[0]

    U = rng.random((c, n))
    U /= np.maximum(U.sum(axis=0, keepdims=True), 1e-12)

    for _ in range(max_iter):
        U_old = U.copy()

        um = U ** m
        centers = (um @ X) / np.maximum(
            um.sum(axis=1, keepdims=True),
            1e-12
        )

        diff = X[None, :, :] - centers[:, None, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        dist = np.maximum(dist, 1e-12)

        power = 2.0 / (m - 1.0)

        for i in range(c):
            ratio = (dist[i:i + 1] / dist) ** power
            U[i] = 1.0 / np.maximum(
                np.sum(ratio, axis=0),
                1e-12
            )

        if np.max(np.abs(U - U_old)) < tol:
            break

    return centers, U

def get_final_membership(features, result_row):

    cluster_num = result_row["cluster_num"]

    if pd.isna(cluster_num):
        raise ValueError("cluster_num is empty.")

    cluster_num = int(round(float(cluster_num)))

    if cluster_num < 2:
        raise ValueError(f"Invalid cluster_num={cluster_num}")

    _, U = fuzzy_cmeans(features, cluster_num)

    peak_idx = result_row.get("peak_index_0", np.nan)

    if pd.notna(peak_idx):
        peak_idx = int(round(float(peak_idx)))

        if 0 <= peak_idx < U.shape[1]:

            final_cluster = int(np.argmax(U[:, peak_idx]))
            return U[final_cluster], final_cluster, cluster_num

    peak_values = np.max(U, axis=1)
    final_cluster = int(np.argmax(peak_values))

    return U[final_cluster], final_cluster, cluster_num

def prepare_record(df, file_id):

    row = get_result_row(df, file_id)

    signal = read_signal(file_id)

    enhanced, kept_idx = enhance_trace(signal)

    if kept_idx.size == 0:
        raise RuntimeError(
            f"Record {file_id}: HOS retained no valid CWT scales."
        )

    features = build_features(enhanced)

    membership, final_cluster, cluster_num = get_final_membership(
        features,
        row
    )

    true_index = int(
        round(
            float(row["true_index_0"])
        )
    )

    if not 0 <= true_index < EXPECTED_SAMPLES:
        raise ValueError(
            f"Record {file_id}: true_index={true_index} is out of range."
        )

    return {
        "file_id": file_id,
        "signal": signal,
        "membership": membership,
        "true_index": true_index,
        "cluster_num": cluster_num,
        "final_cluster": final_cluster
    }

def plot_four_panels(success_data, unreliable_data):

    time_ms = np.arange(EXPECTED_SAMPLES) / FS * 1000.0

    signal_success = success_data["signal"].copy()
    signal_unreliable = unreliable_data["signal"].copy()

    amp1 = np.max(np.abs(signal_success))
    amp2 = np.max(np.abs(signal_unreliable))

    if amp1 > 0:
        signal_success /= amp1

    if amp2 > 0:
        signal_unreliable /= amp2

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.9, 4.1),
        sharex="col"
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    ax_a.plot(
        time_ms,
        signal_success,
        color="0.30",
        linewidth=0.75
    )

    ax_a.axvline(
        success_data["true_index"] / FS * 1000.0,
        color="blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
    )

    ax_b.plot(
        time_ms,
        signal_unreliable,
        color="0.30",
        linewidth=0.75
    )

    ax_b.axvline(
        unreliable_data["true_index"] / FS * 1000.0,
        color="blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
    )

    ax_c.plot(
        time_ms,
        success_data["membership"],
        color="black",
        linewidth=0.8
    )

    ax_c.axvline(
        success_data["true_index"] / FS * 1000.0,
        color="blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
    )

    ax_d.plot(
        time_ms,
        unreliable_data["membership"],
        color="black",
        linewidth=0.8
    )

    ax_d.axvline(
        unreliable_data["true_index"] / FS * 1000.0,
        color="blue",
        linestyle="--",
        linewidth=0.8,
        alpha=0.9,
    )

    ax_a.set_ylabel("Normalized amplitude")
    ax_c.set_ylabel("Membership")

    ax_c.set_xlabel("Time (ms)")
    ax_d.set_xlabel("Time (ms)")

    ax_c.set_ylim(-0.02, 1.02)
    ax_d.set_ylim(-0.02, 1.02)

    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, label in zip(
        [ax_a, ax_b, ax_c, ax_d],
        labels
    ):
        ax.text(
            0.015,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=7.5,

            ha="left",
            va="top"
        )

    for ax in axes.ravel():
        ax.set_xlim(0, 1000)

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=7.5
        )

        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    ax_a.tick_params(labelbottom=False)
    ax_b.tick_params(labelbottom=False)

    ax_b.set_ylabel("")
    ax_d.set_ylabel("")

    fig.tight_layout(
        w_pad=1.5,
        h_pad=0.8
    )

    fig.savefig(
        OUTPUT_TIFF,
        dpi=300,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",

    )

    plt.close(fig)

def main():
    print("=" * 70)
    print("AM-FCM representative field examples")
    print("=" * 70)

    print(f"Reliable case  : Record {SUCCESS_ID}")
    print(f"Ambiguous case : Record {UNRELIABLE_ID}")
    print(f"Feature set    : {'+'.join(FEATURE_SET)}")
    print(f"Sampling rate  : {FS:g} Hz")
    print("Output         : 1200 dpi TIFF")
    print()

    df = load_result_table()

    success_data = prepare_record(
        df,
        SUCCESS_ID
    )

    unreliable_data = prepare_record(
        df,
        UNRELIABLE_ID
    )

    print(
        f"Record {SUCCESS_ID}: "
        f"C={success_data['cluster_num']}, "
        f"final cluster={success_data['final_cluster'] + 1}, "
        f"manual={success_data['true_index'] / FS * 1000:.1f} ms"
    )

    print(
        f"Record {UNRELIABLE_ID}: "
        f"C={unreliable_data['cluster_num']}, "
        f"final cluster={unreliable_data['final_cluster'] + 1}, "
        f"manual={unreliable_data['true_index'] / FS * 1000:.1f} ms"
    )

    plot_four_panels(
        success_data,
        unreliable_data
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Saved to: {OUTPUT_TIFF.resolve()}")

if __name__ == "__main__":
    main()
