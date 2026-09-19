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

from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed, parallel_config

import experiment_common_field as ec

DATA_ROOT = Path("./pre_data")

SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"

RESULT_CSV = Path("./real_amfcm_trace_results.csv")
SUMMARY_CSV = Path("./real_amfcm_summary.csv")

OUT_ROOT = Path("./am_fcm_fig")

OUT_SUCCESS = OUT_ROOT / "success"
OUT_UNRELIABLE = OUT_ROOT / "unreliable"
OUT_ERROR50 = OUT_ROOT / "error_gt_50"
OUT_FAILED = OUT_ROOT / "failed"

FS = float(ec.FS)
DT = 1.0 / FS

EXPECTED_SAMPLES = 1000

LABEL_IS_ONE_BASED = False

SUCCESS_ERROR_LIMIT_MS = 10.0

SUCCESS_IDS = None

MAX_SUCCESS_PLOTS = None

DEFAULT_FEATURE_SET = ("M", "Std")

FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_TOL = 1e-6
RANDOM_STATE = 42

DEFAULT_PEAK_HEIGHT = 0.55
DEFAULT_PEAK_PROMINENCE = 0.12
DEFAULT_PEAK_WIDTH = 2
DEFAULT_PEAK_DISTANCE = 3

N_JOBS = min(
    6,
    max(1, (os.cpu_count() or 2) - 1)
)

def get_feature_set():

    if SUMMARY_CSV.exists():
        try:
            df = pd.read_csv(
                SUMMARY_CSV,
                encoding="utf-8-sig"
            )

            if (
                len(df) > 0
                and "feature_set" in df.columns
            ):
                text = str(
                    df.loc[0, "feature_set"]
                ).strip()

                if text:
                    return tuple(
                        x.strip()
                        for x in text.split("+")
                        if x.strip()
                    )

        except Exception:
            pass

    return DEFAULT_FEATURE_SET

FEATURE_SET = get_feature_set()

def read_signal(file_id):

    path = (
        SIGNAL_DIR /
        f"{int(file_id)}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing signal file: {path}"
        )

    x = np.asarray(
        np.loadtxt(
            path,
            dtype=float
        ),
        dtype=float
    ).ravel()

    if x.size != EXPECTED_SAMPLES:
        raise ValueError(
            f"{path.name}: "
            f"length={x.size}, "
            f"expected={EXPECTED_SAMPLES}"
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            f"{path.name}: contains NaN or Inf"
        )

    return x

def read_label(file_id):

    path = (
        LABEL_DIR /
        f"{int(file_id)}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file: {path}"
        )

    values = np.atleast_1d(
        np.loadtxt(
            path,
            dtype=float
        )
    ).ravel()

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise ValueError(
            f"{path.name}: label is empty"
        )

    label = int(
        round(
            float(
                np.min(values)
            )
        )
    )

    true_idx = (
        label - 1
        if LABEL_IS_ONE_BASED
        else label
    )

    if not (
        0 <= true_idx < EXPECTED_SAMPLES
    ):
        raise ValueError(
            f"{path.name}: "
            f"label={label}, "
            f"converted index={true_idx}"
        )

    return int(true_idx)

def enhance_trace(signal):

    signal = np.asarray(
        signal,
        dtype=float
    ).ravel()

    if hasattr(
        ec,
        "cwt_hos_icwt"
    ):

        enhanced, kept_idx, _ = (
            ec.cwt_hos_icwt(
                signal
            )
        )

        return (
            np.asarray(
                enhanced,
                dtype=float
            ).ravel(),

            np.asarray(
                kept_idx
            ).ravel()
        )

    coeffs, freqs = (
        ec.cwt_morlet_pywt(
            signal,
            ec.DT,
            ec.CWT_FREQUENCIES
        )
    )

    filtered, kept_idx = (
        ec.hos_preprocess_cwt(
            coeffs,
            freqs
        )
    )

    if not hasattr(
        ec,
        "inverse_cwt"
    ):
        raise AttributeError(
            "experiment_common.py"
            " has no inverse_cwt()."
        )

    enhanced = (
        ec.inverse_cwt(
            filtered
        )
    )

    return (
        np.asarray(
            enhanced,
            dtype=float
        ).ravel(),

        np.asarray(
            kept_idx
        ).ravel()
    )

def build_features(enhanced):

    x = np.asarray(
        enhanced,
        dtype=float
    ).ravel()

    feature_bank = {
        "M":
            np.asarray(
                ec.get_amplitude(
                    x,
                    ec.WINDOW_SIZE
                ),
                dtype=float
            ).ravel(),

        "P":
            np.asarray(
                ec.get_energy(
                    x,
                    ec.WINDOW_SIZE
                ),
                dtype=float
            ).ravel(),

        "SLTA":
            np.asarray(
                ec.get_SLTA(
                    x,
                    ec.SHORT_SIZE,
                    ec.LONG_SIZE
                ),
                dtype=float
            ).ravel()
    }

    if "Std" in FEATURE_SET:

        if not hasattr(
            ec,
            "get_std"
        ):
            raise AttributeError(
                "experiment_common.py"
                " has no get_std()."
            )

        feature_bank["Std"] = (
            np.asarray(
                ec.get_std(
                    x,
                    ec.WINDOW_SIZE
                ),
                dtype=float
            ).ravel()
        )

    unknown = [
        name
        for name in FEATURE_SET
        if name not in feature_bank
    ]

    if unknown:
        raise ValueError(
            f"Unsupported feature(s): {unknown}"
        )

    features = (
        np.column_stack(
            [
                feature_bank[name]
                for name in FEATURE_SET
            ]
        )
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        MinMaxScaler()
        .fit_transform(
            features
        )
    )

def fuzzy_cmeans(
    X,
    c,
    m=FCM_M,
    max_iter=FCM_MAX_ITER,
    tol=FCM_TOL,
    random_state=RANDOM_STATE
):

    rng = np.random.default_rng(
        random_state
    )

    X = np.asarray(
        X,
        dtype=float
    )

    n = X.shape[0]

    U = rng.random(
        (c, n)
    )

    U /= np.maximum(
        U.sum(
            axis=0,
            keepdims=True
        ),
        1e-12
    )

    for _ in range(max_iter):

        U_old = U.copy()

        um = U ** m

        centers = (
            um @ X
        ) / np.maximum(
            um.sum(
                axis=1,
                keepdims=True
            ),
            1e-12
        )

        diff = (
            X[None, :, :]
            -
            centers[:, None, :]
        )

        dist = np.sqrt(
            np.sum(
                diff ** 2,
                axis=2
            )
        )

        dist = np.maximum(
            dist,
            1e-12
        )

        power = (
            2.0 /
            (m - 1.0)
        )

        for i in range(c):

            ratio = (
                dist[i:i + 1, :]
                /
                dist
            ) ** power

            U[i] = (
                1.0
                /
                np.maximum(
                    np.sum(
                        ratio,
                        axis=0
                    ),
                    1e-12
                )
            )

        if (
            np.max(
                np.abs(
                    U - U_old
                )
            )
            < tol
        ):
            break

    return centers, U

def get_peak_parameters():

    height = getattr(
        ec,
        "PEAK_HEIGHT",
        DEFAULT_PEAK_HEIGHT
    )

    prominence = getattr(
        ec,
        "PEAK_PROMINENCE",
        getattr(
            ec,
            "PEAK_PROM",
            DEFAULT_PEAK_PROMINENCE
        )
    )

    width = getattr(
        ec,
        "PEAK_WIDTH",
        DEFAULT_PEAK_WIDTH
    )

    distance = getattr(
        ec,
        "PEAK_DISTANCE",
        DEFAULT_PEAK_DISTANCE
    )

    return (
        height,
        prominence,
        width,
        distance
    )

def get_valid_peaks(membership):

    (
        height,
        prominence,
        width,
        distance
    ) = get_peak_parameters()

    peaks, props = find_peaks(
        membership,
        height=height,
        prominence=prominence,
        width=width,
        distance=distance
    )

    return peaks, props

def rerun_amfcm(features):

    result = (
        ec.adaptive_fcm(
            features
        )
    )

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "adaptive_fcm() should return a dict."
        )

    return result

def choose_final_cluster(
    U,
    row,
    rerun_result
):

    n = U.shape[1]

    candidates = [
        row.get(
            "peak_index_0",
            np.nan
        ),

        rerun_result.get(
            "peak",
            np.nan
        ),

        row.get(
            "predicted_index_0",
            np.nan
        ),

        rerun_result.get(
            "arrival",
            np.nan
        )
    ]

    for value in candidates:

        if pd.notna(value):

            idx = int(
                round(
                    float(value)
                )
            )

            if 0 <= idx < n:

                return (
                    int(
                        np.argmax(
                            U[:, idx]
                        )
                    ),
                    idx
                )

    return (
        int(
            np.argmax(
                np.mean(
                    U,
                    axis=1
                )
            )
        ),
        -1
    )

def plot_case(
    file_id,
    category,
    signal,
    true_idx,
    row,
    rerun_result,
    U,
    final_cluster,
    valid_peaks,
    selected_peak,
    output_file
):

    time_ms = (
        np.arange(
            signal.size
        )
        / FS
        * 1000.0
    )

    max_amp = np.max(
        np.abs(signal)
    )

    if max_amp > 0:
        plot_signal = (
            signal / max_amp
        )
    else:
        plot_signal = signal.copy()

    final_u = U[
        final_cluster
    ]

    pred_idx = row.get(
        "predicted_index_0",
        np.nan
    )

    if pd.isna(pred_idx):
        pred_idx = (
            rerun_result.get(
                "arrival",
                np.nan
            )
        )

    cluster_num = row.get(
        "cluster_num",
        np.nan
    )

    if pd.isna(cluster_num):
        cluster_num = (
            rerun_result.get(
                "cluster_num",
                np.nan
            )
        )

    abs_error = row.get(
        "abs_error_ms",
        np.nan
    )

    if (
        pd.isna(abs_error)
        and pd.notna(pred_idx)
    ):
        abs_error = abs(
            (
                float(pred_idx)
                - true_idx
            )
            / FS
            * 1000.0
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6.8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.6, 1]
        }
    )

    ax1, ax2 = axes

    ax1.plot(
        time_ms,
        plot_signal,
        color="0.3",
        lw=0.8,
        label="Waveform"
    )

    ax1.axvline(
        true_idx / FS * 1000.0,
        color="blue",
        lw=1.6,
        label="Manual pick"
    )

    if pd.notna(pred_idx):

        ax1.axvline(
            float(pred_idx)
            / FS
            * 1000.0,
            color="red",
            lw=1.5,
            ls="--",
            label="AM-FCM pick"
        )

    for i, peak in enumerate(
        valid_peaks
    ):

        ax1.axvline(
            peak / FS * 1000.0,
            color="orange",
            lw=1.1,
            ls=":",
            label=(
                "Valid peak"
                if i == 0
                else None
            )
        )

    if pd.notna(
        selected_peak
    ):

        p = int(
            selected_peak
        )

        if 0 <= p < signal.size:

            ax1.plot(
                p / FS * 1000.0,
                plot_signal[p],
                marker="v",
                ms=7,
                color="#1f77b4",
                label="Selected peak"
            )

    ax1.set_ylabel(
        "Normalized amplitude"
    )

    ax1.grid(
        ls="--",
        alpha=0.25
    )

    ax1.legend(
        loc="upper right",
        fontsize=8
    )

    lines = []

    lines.append(
        f"C = "
        f"{int(cluster_num) if pd.notna(cluster_num) else 'NA'}"
    )

    lines.append(
        f"valid peaks = {len(valid_peaks)}"
    )

    lines.append(
        f"Manual pick = "
        f"{true_idx / FS * 1000:.0f} ms"
    )

    if pd.notna(pred_idx):

        lines.append(
            f"AM-FCM pick = "
            f"{float(pred_idx) / FS * 1000:.0f} ms"
        )

        lines.append(
            f"|e| = "
            f"{float(abs_error):.0f} ms"
        )

    else:

        lines.append(
            "No reliable AM-FCM pick"
        )

    ax1.text(
        0.58,
        0.05,
        "\n".join(lines),
        transform=ax1.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(
            facecolor="white",
            edgecolor="0.6",
            alpha=0.9
        )
    )

    ax1.set_title(
        f"Record {file_id} "
        f"({category})",
        fontsize=10
    )

    ax2.plot(
        time_ms,
        final_u,
        color="black",
        lw=1.3,
        label=(
            f"Membership "
            f"(cluster {final_cluster + 1})"
        )
    )

    ax2.axvline(
        true_idx / FS * 1000.0,
        color="blue",
        lw=1.4
    )

    if pd.notna(pred_idx):

        ax2.axvline(
            float(pred_idx)
            / FS
            * 1000.0,
            color="red",
            lw=1.4,
            ls="--"
        )

    for peak in valid_peaks:

        ax2.axvline(
            peak / FS * 1000.0,
            color="orange",
            lw=1.0,
            ls=":"
        )

    if pd.notna(
        selected_peak
    ):

        p = int(
            selected_peak
        )

        if 0 <= p < len(final_u):

            ax2.plot(
                p / FS * 1000.0,
                final_u[p],
                marker="v",
                ms=7,
                color="#1f77b4"
            )

    ax2.set_xlabel(
        "Time (ms)"
    )

    ax2.set_ylabel(
        "Membership"
    )

    ax2.set_ylim(
        -0.02,
        1.02
    )

    ax2.grid(
        ls="--",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

def process_one(
    file_id,
    category,
    result_map
):

    try:

        row = result_map.get(
            int(file_id),
            {}
        )

        signal = read_signal(
            file_id
        )

        true_idx = read_label(
            file_id
        )

        enhanced, kept_idx = (
            enhance_trace(
                signal
            )
        )

        if kept_idx.size == 0:
            raise RuntimeError(
                "HOS retained no valid scales."
            )

        features = (
            build_features(
                enhanced
            )
        )

        rerun_result = (
            rerun_amfcm(
                features
            )
        )

        cluster_num = row.get(
            "cluster_num",
            np.nan
        )

        if pd.isna(
            cluster_num
        ):
            cluster_num = (
                rerun_result.get(
                    "cluster_num",
                    np.nan
                )
            )

        if pd.isna(
            cluster_num
        ):
            raise ValueError(
                "Unable to obtain cluster_num."
            )

        cluster_num = int(
            round(
                float(
                    cluster_num
                )
            )
        )

        centers, U = (
            fuzzy_cmeans(
                features,
                cluster_num
            )
        )

        final_cluster, _ = (
            choose_final_cluster(
                U,
                row,
                rerun_result
            )
        )

        final_u = U[
            final_cluster
        ]

        valid_peaks, _ = (
            get_valid_peaks(
                final_u
            )
        )

        selected_peak = (
            row.get(
                "peak_index_0",
                np.nan
            )
        )

        if pd.isna(
            selected_peak
        ):
            selected_peak = (
                rerun_result.get(
                    "peak",
                    np.nan
                )
            )

        if (
            pd.isna(
                selected_peak
            )
            and len(
                valid_peaks
            ) > 0
        ):
            selected_peak = int(
                valid_peaks[
                    np.argmax(
                        final_u[
                            valid_peaks
                        ]
                    )
                ]
            )

        if category == "success":
            out_dir = OUT_SUCCESS

        elif category == "unreliable":
            out_dir = OUT_UNRELIABLE

        elif category == "error_gt_50":
            out_dir = OUT_ERROR50

        else:
            out_dir = OUT_FAILED

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            out_dir /
            f"{int(file_id)}.png"
        )

        plot_case(
            file_id=file_id,
            category=category,
            signal=signal,
            true_idx=true_idx,
            row=row,
            rerun_result=rerun_result,
            U=U,
            final_cluster=final_cluster,
            valid_peaks=valid_peaks,
            selected_peak=selected_peak,
            output_file=output_file
        )

        return {
            "file_id":
                int(file_id),

            "category":
                category,

            "plotted":
                True,

            "cluster_num":
                cluster_num,

            "final_cluster":
                final_cluster + 1,

            "valid_peak_count":
                len(valid_peaks),

            "selected_peak":
                selected_peak,

            "output_file":
                str(output_file),

            "message":
                ""
        }

    except Exception as error:

        return {
            "file_id":
                int(file_id),

            "category":
                category,

            "plotted":
                False,

            "cluster_num":
                np.nan,

            "final_cluster":
                np.nan,

            "valid_peak_count":
                np.nan,

            "selected_peak":
                np.nan,

            "output_file":
                "",

            "message":
                str(error)
        }

def build_tasks(df):

    df = df.copy()

    df["file_id"] = (
        df["file_id"]
        .astype(int)
    )

    success_flag = (
        df["success"]
        .fillna(False)
        .astype(bool)
    )

    success_df = df[
        success_flag
        &
        df["abs_error_ms"].notna()
        &
        (
            df["abs_error_ms"]
            <= SUCCESS_ERROR_LIMIT_MS
        )
    ].copy()

    if SUCCESS_IDS is not None:

        success_ids = set(
            int(x)
            for x in SUCCESS_IDS
        )

        success_df = (
            success_df[
                success_df["file_id"]
                .isin(success_ids)
            ]
        )

    if (
        MAX_SUCCESS_PLOTS
        is not None
    ):

        success_df = (
            success_df
            .sort_values(
                "abs_error_ms"
            )
            .head(
                MAX_SUCCESS_PLOTS
            )
        )

    unreliable_df = df[
        df["status"]
        .astype(str)
        .str.strip()
        .eq("Unreliable")
    ].copy()

    error50_df = df[
        success_flag
        &
        df["abs_error_ms"].notna()
        &
        (
            df["abs_error_ms"]
            > 50.0
        )
    ].copy()

    failed_df = df[
        df["status"]
        .astype(str)
        .str.strip()
        .eq("Failed")
    ].copy()

    tasks = []

    for fid in success_df["file_id"]:
        tasks.append(
            (
                int(fid),
                "success"
            )
        )

    for fid in unreliable_df["file_id"]:
        tasks.append(
            (
                int(fid),
                "unreliable"
            )
        )

    for fid in error50_df["file_id"]:
        tasks.append(
            (
                int(fid),
                "error_gt_50"
            )
        )

    for fid in failed_df["file_id"]:
        tasks.append(
            (
                int(fid),
                "failed"
            )
        )

    return (
        tasks,
        success_df,
        unreliable_df,
        error50_df,
        failed_df
    )

def main():

    for directory in [
        OUT_ROOT,
        OUT_SUCCESS,
        OUT_UNRELIABLE,
        OUT_ERROR50,
        OUT_FAILED
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    if not RESULT_CSV.exists():
        raise FileNotFoundError(
            f"Result files not found: "
            f"{RESULT_CSV}"
        )

    df = pd.read_csv(
        RESULT_CSV,
        encoding="utf-8-sig"
    )

    result_map = {
        int(row["file_id"]):
            row.to_dict()

        for _, row
        in df.iterrows()
    }

    (
        tasks,
        success_df,
        unreliable_df,
        error50_df,
        failed_df
    ) = build_tasks(df)

    success_df.to_csv(
        OUT_ROOT /
        "success_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    unreliable_df.to_csv(
        OUT_ROOT /
        "unreliable_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    error50_df.to_csv(
        OUT_ROOT /
        "error_gt_50_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    failed_df.to_csv(
        OUT_ROOT /
        "failed_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 72)
    print("AM-FCM field-case plotting")
    print("=" * 72)

    print(
        "Feature set:",
        "+".join(
            FEATURE_SET
        )
    )

    print(
        "Success criterion:",
        f"|error| <= "
        f"{SUCCESS_ERROR_LIMIT_MS:g} ms"
    )

    print(
        f"Success      : "
        f"{len(success_df)}"
    )

    print(
        f"Unreliable   : "
        f"{len(unreliable_df)}"
    )

    print(
        f"Error >50 ms : "
        f"{len(error50_df)}"
    )

    print(
        f"Failed       : "
        f"{len(failed_df)}"
    )

    print(
        f"Total plots  : "
        f"{len(tasks)}"
    )

    print()

    with parallel_config(
        backend="loky",
        inner_max_num_threads=1
    ):

        rows = Parallel(
            n_jobs=N_JOBS,
            verbose=10
        )(
            delayed(
                process_one
            )(
                file_id,
                category,
                result_map
            )

            for (
                file_id,
                category
            )
            in tasks
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        OUT_ROOT /
        "plot_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plotted = result_df[
        result_df["plotted"]
        ==
        True
    ]

    skipped = result_df[
        result_df["plotted"]
        ==
        False
    ]

    print()
    print("=" * 72)
    print("Finished")
    print("=" * 72)

    print(
        "Successfully plotted:",
        len(plotted)
    )

    print(
        "Skipped:",
        len(skipped)
    )

    if len(skipped):

        print(
            "\nSkipped records:"
        )

        for row in (
            skipped.itertuples(
                index=False
            )
        ):

            print(
                f"{row.file_id} "
                f"({row.category}): "
                f"{row.message}"
            )

    print()
    print("Output:")
    print("Success     :", OUT_SUCCESS.resolve())
    print("Unreliable  :", OUT_UNRELIABLE.resolve())
    print("Error >50 ms:", OUT_ERROR50.resolve())
    print("Failed      :", OUT_FAILED.resolve())

if __name__ == "__main__":
    main()
