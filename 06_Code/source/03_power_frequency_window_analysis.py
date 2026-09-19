"""
Power-window / dominant-frequency PC analysis
=============================================

Purpose
-------
This script is intended for the feature-analysis section before AM-FCM.

It DOES NOT call AM-FCM.

For every synthetic noisy trace:

    noisy trace
        -> CWT
        -> calibrated HOS scale selection
        -> numerical iCWT from retained COMPLEX CWT coefficients
        -> local Power P using a candidate WINDOW_SIZE
        -> Pearson correlation with an AIC-derived binary signal label

The binary reference is constructed from the NOISE-FREE clean waveform:
    0 = background/noise region
    1 = microseismic-signal interval

The signal interval is defined by:
    start = AIC change point on the pre-peak branch
    end   = AIC change point on the reversed post-peak branch

The reported PC follows the feature-representation convention:
    PC = |Pearson(P, binary_label)|

Experimental grid
-----------------
WINDOW_SIZES = [5, 9, 13, 17, 21, 25, 31]
FEATURE       = Power only
FREQUENCIES   = [50, 100, 150, 200, 250, 300, 350, 400] Hz
SNR           = -8 dB
WAVEFORM      = Ricker
NOISE         = WGN
REPEATS       = 200 per frequency

Strict pairing
--------------
For a given frequency and repeat:
    - one noisy trace is generated;
    - one calibrated-HOS iCWT representation is obtained;
    - every candidate window uses the SAME reconstructed trace and SAME label.

Outputs
-------
results_pc_power_frequency_window/
    raw_pc_results.csv
    summary_pc_by_frequency_window.csv
    best_window_by_frequency.csv
    best_window_frequency_trend.csv
    run_configuration.txt
    figures/
        pc_window_curves.png
        pc_frequency_window_heatmap.png
        best_window_vs_frequency.png

Important interpretation
------------------------
This experiment answers:
    "Does the preferred local feature window vary with dominant frequency?"

It does NOT by itself select the final multi-feature AM-FCM input.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "./.mpl-cache")

import sys
from pathlib import Path

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pywt

from joblib import Parallel, delayed, parallel_config
from scipy.stats import kurtosis, spearmanr

parent_dir = str(Path.cwd().resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import experiment_common as ec

WINDOW_SIZES = np.arange(3, 41, 2).tolist()
FEATURE = "P"

FREQUENCIES = np.arange(50, 460, 10).tolist()

SNR_VALUES = [
    -5,
]

WAVEFORM = "ricker"
NOISE_TYPE = "WGN"

N_REPEAT = 100
BASE_SEED = 20260811

CPU_COUNT = os.cpu_count() or 4
N_JOBS = min(
    32,
    max(
        1,
        CPU_COUNT - 2,
    ),
)

FS = float(ec.FS)
DT = float(ec.DT)

WAVELET_NAME = getattr(
    ec,
    "CWT_WAVELET",
    "cmor1.5-1.0",
)

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    SCRIPT_DIR
    / "results_pc_power_frequency_window"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

CALIBRATED_HOS_FILE = None

AIC_MIN_SIDE = 2
_AIC_EPS = 1e-12

def _aic_change_point(
    signal,
    min_side=AIC_MIN_SIDE,
):
    """
    Return the AIC change-point index within a 1-D noise-free waveform.

    AIC(k) =
        k * log(var(x[:k]))
        + (N-k) * log(var(x[k:]))

    The search excludes `min_side` samples at each edge.
    """
    x = np.asarray(
        signal,
        dtype=np.float64,
    ).ravel()

    n = x.size

    if n < 2 * min_side + 1:
        raise ValueError(
            f"AIC input is too short: n={n}, min_side={min_side}."
        )

    if not np.all(
        np.isfinite(
            x
        )
    ):
        raise ValueError(
            "AIC input contains NaN or infinite values."
        )

    cumulative_sum = np.cumsum(
        x
    )

    cumulative_square_sum = np.cumsum(
        x * x
    )

    k = np.arange(
        min_side,
        n - min_side,
        dtype=np.int64,
    )

    n_left = k.astype(
        np.float64
    )

    sum_left = cumulative_sum[
        k - 1
    ]

    square_sum_left = cumulative_square_sum[
        k - 1
    ]

    var_left = (
        square_sum_left
        / n_left
        - (
            sum_left
            / n_left
        )
        ** 2
    )

    n_right = (
        n - k
    ).astype(
        np.float64
    )

    sum_right = (
        cumulative_sum[-1]
        - sum_left
    )

    square_sum_right = (
        cumulative_square_sum[-1]
        - square_sum_left
    )

    var_right = (
        square_sum_right
        / n_right
        - (
            sum_right
            / n_right
        )
        ** 2
    )

    var_left = np.maximum(
        var_left,
        _AIC_EPS,
    )

    var_right = np.maximum(
        var_right,
        _AIC_EPS,
    )

    score = (
        n_left
        * np.log(
            var_left
        )
        + n_right
        * np.log(
            var_right
        )
    )

    return int(
        k[
            np.argmin(
                score
            )
        ]
    )

def aic_signal_bounds(
    clean_wavelet,
    min_side=AIC_MIN_SIDE,
):
    """
    Determine a continuous signal interval from a noise-free waveform.

    The same AIC change-point criterion is used for both boundaries, but
    the known peak of the noise-free synthetic waveform is used ONLY to
    separate the leading and trailing branches:

        onset:
            AIC on the support from its left edge through the clean peak.

        end:
            AIC on the reversed support from its right edge through the
            same clean peak, then mapped back to forward time.

    Why this constraint is required
    -------------------------------
    A complete Ricker waveform contains two transitions:
        background -> waveform -> background.

    If forward and reverse AIC are each applied to the entire waveform,
    either search can lock onto the opposite transition. For some dominant
    frequencies this produces crossing candidates (end < onset).

    Splitting at the known noise-free peak prevents this ambiguity. The peak
    is NOT used as a label boundary; it only delimits the two AIC searches.
    """
    x = np.asarray(
        clean_wavelet,
        dtype=np.float64,
    ).ravel()

    if x.size == 0:
        raise ValueError(
            "clean_wavelet is empty."
        )

    if not np.all(
        np.isfinite(
            x
        )
    ):
        raise ValueError(
            "clean_wavelet contains NaN or infinite values."
        )

    if (
        np.max(
            np.abs(
                x
            )
        )
        <= np.finfo(
            float
        ).eps
    ):
        raise ValueError(
            "clean_wavelet has zero amplitude."
        )

    nonzero = np.flatnonzero(
        x != 0.0
    )

    if (
        nonzero.size > 0
        and nonzero.size < x.size
    ):
        support_start = int(
            nonzero[0]
        )
        support_end = int(
            nonzero[-1]
        ) + 1
    else:
        support_start = 0
        support_end = x.size

    local = x[
        support_start:
        support_end
    ]

    if (
        local.size
        < 2 * min_side + 1
    ):
        raise ValueError(
            "AIC waveform support is too short: "
            f"{local.size} samples."
        )

    peak_local = int(
        np.argmax(
            np.abs(
                local
            )
        )
    )

    leading = local[
        :peak_local + 1
    ]

    if (
        leading.size
        < 2 * min_side + 1
    ):
        raise ValueError(
            "Pre-peak waveform branch is too short for AIC: "
            f"{leading.size} samples."
        )

    onset_local = _aic_change_point(
        leading,
        min_side=min_side,
    )

    trailing = local[
        peak_local:
    ]

    if (
        trailing.size
        < 2 * min_side + 1
    ):
        raise ValueError(
            "Post-peak waveform branch is too short for AIC: "
            f"{trailing.size} samples."
        )

    reverse_pick = _aic_change_point(
        trailing[::-1],
        min_side=min_side,
    )

    trailing_end_offset = (
        trailing.size
        - 1
        - reverse_pick
    )

    end_local = (
        peak_local
        + trailing_end_offset
    )

    onset = (
        support_start
        + onset_local
    )

    peak = (
        support_start
        + peak_local
    )

    end = (
        support_start
        + end_local
    )

    if not (
        onset
        < peak
        < end
    ):
        raise RuntimeError(
            "Peak-constrained AIC produced invalid bounds: "
            f"onset={onset}, peak={peak}, end={end}, "
            f"support=[{support_start}, {support_end})."
        )

    return (
        int(
            onset
        ),
        int(
            end
        ),
    )

def build_binary_labels(
    clean_wavelet
):
    x = np.asarray(
        clean_wavelet
    ).ravel()

    onset, end = (
        aic_signal_bounds(
            x
        )
    )

    labels = np.zeros(
        x.size,
        dtype=np.int8,
    )

    labels[
        onset:
        end + 1
    ] = 1

    return (
        labels,
        onset,
        end,
    )

def validate_common():
    required = [
        "FS",
        "DT",
        "CWT_FREQUENCIES",
        "cwt_morlet_pywt",
        "simulate",
    ]

    missing = [
        name
        for name
        in required
        if not hasattr(
            ec,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "experiment_common.py does not provide the required interface. "
            f"Missing: {missing}"
        )

    cwt_frequencies = np.asarray(
        ec.CWT_FREQUENCIES,
        dtype=float,
    ).ravel()

    if cwt_frequencies.size < 2:
        raise RuntimeError(
            "CWT_FREQUENCIES is invalid."
        )

    if not np.all(
        np.isfinite(
            cwt_frequencies
        )
    ):
        raise RuntimeError(
            "CWT_FREQUENCIES contains non-finite values."
        )

    if any(
        int(
            window
        ) <= 0
        or int(
            window
        ) % 2 == 0
        for window
        in WINDOW_SIZES
    ):
        raise ValueError(
            "All WINDOW_SIZES must be positive odd integers."
        )

    return (
        cwt_frequencies
    )

CWT_FREQUENCIES = (
    validate_common()
)

def calibration_from_common():
    threshold_names = [
        "HOS_CALIBRATED_THRESHOLDS",
        "CALIBRATED_HOS_THRESHOLDS",
    ]

    frequency_names = [
        "HOS_CALIBRATION_FREQUENCIES",
        "CALIBRATED_HOS_FREQUENCIES",
    ]

    thresholds = None
    frequencies = None
    threshold_name = ""
    frequency_name = ""

    for name in threshold_names:
        if hasattr(
            ec,
            name,
        ):
            value = getattr(
                ec,
                name,
            )

            if value is None:
                continue

            value = np.asarray(
                value,
                dtype=float,
            ).ravel()

            if (
                value.size
                == CWT_FREQUENCIES.size
            ):
                thresholds = value
                threshold_name = name
                break

    if thresholds is None:
        return None

    for name in frequency_names:
        if hasattr(
            ec,
            name,
        ):
            value = getattr(
                ec,
                name,
            )

            if value is None:
                continue

            value = np.asarray(
                value,
                dtype=float,
            ).ravel()

            if (
                value.size
                == CWT_FREQUENCIES.size
            ):
                frequencies = value
                frequency_name = name
                break

    if frequencies is None:
        frequencies = (
            CWT_FREQUENCIES.copy()
        )
        frequency_name = (
            "experiment_common.CWT_FREQUENCIES"
        )

    if not np.allclose(
        frequencies,
        CWT_FREQUENCIES,
        rtol=1e-8,
        atol=1e-10,
    ):
        return None

    if (
        np.any(
            ~np.isfinite(
                thresholds
            )
        )
        or np.any(
            thresholds <= 0
        )
    ):
        return None

    return {
        "thresholds": thresholds,
        "frequencies": frequencies,
        "source": (
            f"experiment_common.py: "
            f"{threshold_name}; {frequency_name}"
        ),
        "csv_path": None,
    }

def calibration_candidates():
    if (
        CALIBRATED_HOS_FILE
        is not None
    ):
        return [
            Path(
                CALIBRATED_HOS_FILE
            )
        ]

    candidates = [
        SCRIPT_DIR
        / "calibrated_hos_thresholds.csv",

        SCRIPT_DIR.parent
        / "calibrated_hos_thresholds.csv",
    ]

    candidates.extend(
        sorted(
            SCRIPT_DIR.glob(
                "calibrated_hos*.csv"
            )
        )
    )

    candidates.extend(
        sorted(
            SCRIPT_DIR.parent.glob(
                "calibrated_hos*.csv"
            )
        )
    )

    unique = []
    seen = set()

    for path in candidates:
        key = (
            str(
                path.resolve()
            )
            if path.exists()
            else str(
                path
            )
        )

        if key not in seen:
            seen.add(
                key
            )
            unique.append(
                path
            )

    return (
        unique
    )

def calibration_from_csv():
    checked = []

    for path in calibration_candidates():
        checked.append(
            str(
                path
            )
        )

        if not path.exists():
            continue

        try:
            frame = pd.read_csv(
                path
            )
        except Exception:
            continue

        required = {
            "frequency_hz",
            "calibrated_threshold_q90_absK",
        }

        if not required.issubset(
            frame.columns
        ):
            continue

        frame = (
            frame.copy()
        )

        if (
            "scale_index"
            in frame.columns
        ):
            frame = (
                frame.sort_values(
                    "scale_index"
                )
            )
        else:
            frame = (
                frame.sort_values(
                    "frequency_hz"
                )
            )

        frequencies = pd.to_numeric(
            frame[
                "frequency_hz"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        thresholds = pd.to_numeric(
            frame[
                "calibrated_threshold_q90_absK"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                frequencies
            )
            & np.isfinite(
                thresholds
            )
        )

        frequencies = (
            frequencies[
                valid
            ]
        )

        thresholds = (
            thresholds[
                valid
            ]
        )

        if (
            frequencies.size
            != CWT_FREQUENCIES.size
            or thresholds.size
            != CWT_FREQUENCIES.size
        ):
            continue

        if not np.allclose(
            frequencies,
            CWT_FREQUENCIES,
            rtol=1e-8,
            atol=1e-10,
        ):
            continue

        if np.any(
            thresholds <= 0
        ):
            continue

        return {
            "thresholds": thresholds,
            "frequencies": frequencies,
            "source": (
                f"CSV: "
                f"{path.resolve()}"
            ),
            "csv_path": str(
                path.resolve()
            ),
        }

    raise FileNotFoundError(
        "No compatible calibrated-HOS threshold file was found.\n"
        "Checked:\n  - "
        + "\n  - ".join(
            checked
        )
    )

CALIBRATION = (
    calibration_from_common()
)

if CALIBRATION is None:
    CALIBRATION = (
        calibration_from_csv()
    )

CALIBRATED_THRESHOLDS = np.asarray(
    CALIBRATION[
        "thresholds"
    ],
    dtype=float,
).ravel()

CALIBRATION_FREQUENCIES = np.asarray(
    CALIBRATION[
        "frequencies"
    ],
    dtype=float,
).ravel()

def get_analysis_scales():
    wavelet = (
        pywt.ContinuousWavelet(
            WAVELET_NAME
        )
    )

    normalized_frequencies = (
        CWT_FREQUENCIES
        * DT
    )

    scales = (
        pywt.frequency2scale(
            wavelet,
            normalized_frequencies,
        )
    )

    scales = np.asarray(
        scales,
        dtype=float,
    ).ravel()

    if (
        scales.size
        != CWT_FREQUENCIES.size
    ):
        raise RuntimeError(
            "Unexpected CWT scale count."
        )

    return scales

ANALYSIS_SCALES = (
    get_analysis_scales()
)

def inverse_cwt_shape(
    coefficients,
    scales=ANALYSIS_SCALES,
):
    coefficients = np.asarray(
        coefficients
    )

    scales = np.asarray(
        scales,
        dtype=float,
    ).ravel()

    if coefficients.ndim != 2:
        raise ValueError(
            "CWT coefficients must be 2-D."
        )

    if (
        coefficients.shape[
            0
        ]
        != scales.size
    ):
        raise ValueError(
            "Scale count does not match CWT rows."
        )

    order = np.argsort(
        scales
    )

    scales_sorted = (
        scales[
            order
        ]
    )

    coefficients_sorted = (
        coefficients[
            order,
            :
        ]
    )

    integrand = (
        np.real(
            coefficients_sorted
        )
        / (
            scales_sorted[
                :,
                None
            ]
            ** 1.5
        )
    )

    if hasattr(
        np,
        "trapezoid",
    ):
        reconstructed = (
            np.trapezoid(
                integrand,
                x=scales_sorted,
                axis=0,
            )
        )
    else:
        reconstructed = (
            np.trapz(
                integrand,
                x=scales_sorted,
                axis=0,
            )
        )

    reconstructed = np.asarray(
        reconstructed,
        dtype=float,
    ).ravel()

    np.nan_to_num(
        reconstructed,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        reconstructed
    )

def real_cwt_kurtosis(
    coefficients
):
    values = np.asarray(
        kurtosis(
            np.real(
                np.asarray(
                    coefficients
                )
            ),
            axis=1,
            fisher=True,
            bias=True,
            nan_policy="omit",
        ),
        dtype=float,
    )

    np.nan_to_num(
        values,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        values
    )

def calibrated_hos_icwt(
    noisy
):
    noisy = np.asarray(
        noisy,
        dtype=float,
    ).ravel()

    (
        coefficients,
        actual_frequencies,
    ) = ec.cwt_morlet_pywt(
        noisy,
        DT,
        CWT_FREQUENCIES,
    )

    coefficients = np.asarray(
        coefficients
    )

    actual_frequencies = np.asarray(
        actual_frequencies,
        dtype=float,
    ).ravel()

    if coefficients.ndim != 2:
        raise ValueError(
            f"Unexpected CWT shape: "
            f"{coefficients.shape}"
        )

    if (
        coefficients.shape[
            0
        ]
        != CWT_FREQUENCIES.size
    ):
        raise ValueError(
            "CWT scale count does not match calibration."
        )

    if not np.allclose(
        actual_frequencies,
        CALIBRATION_FREQUENCIES,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Actual CWT frequency grid does not match "
            "the calibrated-HOS frequency grid."
        )

    k_values = (
        real_cwt_kurtosis(
            coefficients
        )
    )

    mask = (
        np.abs(
            k_values
        )
        > CALIBRATED_THRESHOLDS
    )

    selected = np.zeros_like(
        coefficients
    )

    selected[
        mask,
        :
    ] = coefficients[
        mask,
        :
    ]

    if np.any(
        mask
    ):
        reconstructed = (
            inverse_cwt_shape(
                selected,
                ANALYSIS_SCALES,
            )
        )
    else:
        reconstructed = np.zeros(
            noisy.size,
            dtype=float,
        )

    return {
        "series": reconstructed,
        "retained_scale_count": int(
            mask.sum()
        ),
        "retained_scale_fraction": float(
            mask.mean()
        ),
    }

def local_power(
    signal,
    window_size,
):
    """
    Local sum of squares using the same centered-window convention used
    in the previous feature experiments.
    """
    x = np.asarray(
        signal,
        dtype=float,
    ).ravel()

    window_size = int(
        window_size
    )

    half = (
        window_size
        // 2
    )

    n = x.size

    indices = np.arange(
        n,
        dtype=int,
    )

    starts = np.maximum(
        0,
        indices - half,
    )

    ends = np.minimum(
        n,
        indices + half + 1,
    )

    x2 = (
        x ** 2
    )

    cumulative = np.concatenate(
        (
            np.array(
                [0.0]
            ),
            np.cumsum(
                x2,
                dtype=float,
            ),
        )
    )

    power = (
        cumulative[
            ends
        ]
        - cumulative[
            starts
        ]
    )

    np.nan_to_num(
        power,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        power
    )

def pearson_pc(
    feature,
    labels,
):
    x = np.asarray(
        feature,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        labels,
        dtype=np.float64,
    ).ravel()

    if x.size != y.size:
        raise ValueError(
            f"Feature/label length mismatch: "
            f"{x.size} vs {y.size}."
        )

    if (
        np.std(
            x
        )
        <= np.finfo(
            float
        ).eps
    ):
        return (
            np.nan,
            np.nan,
        )

    if (
        np.std(
            y
        )
        <= np.finfo(
            float
        ).eps
    ):
        return (
            np.nan,
            np.nan,
        )

    r = float(
        np.corrcoef(
            x,
            y,
        )[
            0,
            1
        ]
    )

    return (
        r,
        abs(
            r
        ),
    )

def make_rng(
    frequency_index,
    snr_index,
    repeat,
):
    seed = np.random.SeedSequence(
        [
            BASE_SEED,
            int(
                frequency_index
            ),
            int(
                snr_index
            ),
            int(
                repeat
            ),
        ]
    )

    return (
        np.random.default_rng(
            seed
        )
    )

def run_one_record(
    frequency,
    snr_db,
    repeat,
):
    frequency_index = (
        FREQUENCIES.index(
            frequency
        )
    )

    snr_index = (
        SNR_VALUES.index(
            snr_db
        )
    )

    rng = make_rng(
        frequency_index,
        snr_index,
        repeat,
    )

    (
        noisy,
        clean,
        metadata,
    ) = ec.simulate(
        waveform=WAVEFORM,
        frequency=frequency,
        snr_db=snr_db,
        noise_type=NOISE_TYPE,
        noise_bank=None,
        rng=rng,
    )

    noisy = np.asarray(
        noisy,
        dtype=float,
    ).ravel()

    clean = np.asarray(
        clean,
        dtype=float,
    ).ravel()

    (
        labels,
        label_onset,
        label_end,
    ) = build_binary_labels(
        clean
    )

    preprocessing = (
        calibrated_hos_icwt(
            noisy
        )
    )

    reconstructed = (
        preprocessing[
            "series"
        ]
    )

    rows = []

    for window_size in WINDOW_SIZES:
        power = local_power(
            reconstructed,
            window_size,
        )

        pearson_r, pc_abs = (
            pearson_pc(
                power,
                labels,
            )
        )

        row = {
            "waveform": WAVEFORM,
            "noise_type": NOISE_TYPE,
            "frequency_hz": int(
                frequency
            ),
            "snr_db": float(
                snr_db
            ),
            "repeat": int(
                repeat
            ),
            "preprocessing": (
                "CWT-calibrated_HOS-iCWT"
            ),
            "feature": FEATURE,
            "window_size": int(
                window_size
            ),
            "pearson_r": (
                pearson_r
            ),
            "pc_abs": (
                pc_abs
            ),
            "label_onset": int(
                label_onset
            ),
            "label_end": int(
                label_end
            ),
            "label_width_samples": int(
                label_end
                - label_onset
                + 1
            ),
            "label_fraction": float(
                np.mean(
                    labels
                )
            ),
            "retained_scale_count": int(
                preprocessing[
                    "retained_scale_count"
                ]
            ),
            "retained_scale_fraction": float(
                preprocessing[
                    "retained_scale_fraction"
                ]
            ),
        }

        for key in [
            "true_arrival",
            "actual_snr_db",
            "reference_method",
            "true_peak_sample",
        ]:
            if key in metadata:
                row[
                    key
                ] = metadata[
                    key
                ]

        rows.append(
            row
        )

    return (
        rows
    )

def summarize_pc(
    raw
):
    summary = (
        raw.groupby(
            [
                "frequency_hz",
                "snr_db",
                "window_size",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            n=(
                "pc_abs",
                "count",
            ),
            mean_pc=(
                "pc_abs",
                "mean",
            ),
            std_pc=(
                "pc_abs",
                "std",
            ),
            median_pc=(
                "pc_abs",
                "median",
            ),
            min_pc=(
                "pc_abs",
                "min",
            ),
            max_pc=(
                "pc_abs",
                "max",
            ),
            mean_signed_r=(
                "pearson_r",
                "mean",
            ),
            mean_label_width_samples=(
                "label_width_samples",
                "mean",
            ),
            mean_retained_scale_count=(
                "retained_scale_count",
                "mean",
            ),
        )
        .reset_index()
    )

    return (
        summary
    )

def best_window_table(
    summary
):
    rows = []

    for (
        frequency,
        group,
    ) in summary.groupby(
        "frequency_hz",
        sort=True,
    ):
        ranked = (
            group.sort_values(
                by=[
                    "mean_pc",
                    "std_pc",
                    "window_size",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        best = (
            ranked.iloc[
                0
            ]
        )

        max_pc = float(
            best[
                "mean_pc"
            ]
        )

        threshold = (
            0.99
            * max_pc
        )

        near = (
            group[
                group[
                    "mean_pc"
                ]
                >= threshold
            ][
                "window_size"
            ]
            .astype(
                int
            )
            .sort_values()
            .tolist()
        )

        rows.append(
            {
                "frequency_hz": int(
                    frequency
                ),
                "best_window_size": int(
                    best[
                        "window_size"
                    ]
                ),
                "best_mean_pc": float(
                    best[
                        "mean_pc"
                    ]
                ),
                "best_std_pc": float(
                    best[
                        "std_pc"
                    ]
                ),
                "near_optimal_windows_99pct": (
                    "|".join(
                        str(
                            value
                        )
                        for value
                        in near
                    )
                ),
                "mean_label_width_samples": float(
                    best[
                        "mean_label_width_samples"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )

def frequency_window_trend(
    best
):
    x = best[
        "frequency_hz"
    ].to_numpy(
        dtype=float
    )

    y = best[
        "best_window_size"
    ].to_numpy(
        dtype=float
    )

    if (
        x.size >= 3
        and np.unique(
            y
        ).size > 1
    ):
        rho, pvalue = (
            spearmanr(
                x,
                y
            )
        )
    else:
        rho = np.nan
        pvalue = np.nan

    return pd.DataFrame(
        [
            {
                "n_frequency_levels": int(
                    x.size
                ),
                "spearman_rho_frequency_vs_best_window": float(
                    rho
                ),
                "spearman_pvalue": float(
                    pvalue
                ),
                "interpretation_note": (
                    "Negative rho indicates that the preferred "
                    "window tends to decrease as dominant frequency increases. "
                    "Because best windows are selected from a discrete grid, "
                    "the curve/heatmap should remain the primary evidence."
                ),
            }
        ]
    )

def plot_pc_window_curves(
    summary
):
    fig, ax = plt.subplots(
        figsize=(
            8.8,
            5.4,
        )
    )

    markers = [
        "o",
        "s",
        "^",
        "v",
        "D",
        "P",
        "X",
        ">",
    ]

    for (
        index,
        frequency,
    ) in enumerate(
        FREQUENCIES
    ):
        subset = (
            summary[
                summary[
                    "frequency_hz"
                ]
                == frequency
            ]
            .sort_values(
                "window_size"
            )
        )

        ax.plot(
            subset[
                "window_size"
            ],
            subset[
                "mean_pc"
            ],
            marker=markers[
                index
                % len(
                    markers
                )
            ],
            linewidth=1.4,
            markersize=5,
            label=(
                f"{frequency} Hz"
            ),
        )

    ax.set_xlabel(
        "Window size (samples)"
    )

    ax.set_ylabel(
        "Absolute Pearson correlation, |r|"
    )

    ax.set_xticks(
        WINDOW_SIZES
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax.legend(
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "pc_window_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

def plot_pc_heatmap(
    summary
):
    matrix = np.full(
        (
            len(
                FREQUENCIES
            ),
            len(
                WINDOW_SIZES
            ),
        ),
        np.nan,
        dtype=float,
    )

    for (
        i,
        frequency,
    ) in enumerate(
        FREQUENCIES
    ):
        for (
            j,
            window_size,
        ) in enumerate(
            WINDOW_SIZES
        ):
            values = summary[
                (
                    summary[
                        "frequency_hz"
                    ]
                    == frequency
                )
                & (
                    summary[
                        "window_size"
                    ]
                    == window_size
                )
            ][
                "mean_pc"
            ]

            if not values.empty:
                matrix[
                    i,
                    j
                ] = float(
                    values.iloc[
                        0
                    ]
                )

    fig, ax = plt.subplots(
        figsize=(
            8.2,
            5.0,
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(
            len(
                WINDOW_SIZES
            )
        )
    )

    ax.set_xticklabels(
        [
            str(
                value
            )
            for value
            in WINDOW_SIZES
        ]
    )

    ax.set_yticks(
        np.arange(
            len(
                FREQUENCIES
            )
        )
    )

    ax.set_yticklabels(
        [
            str(
                value
            )
            for value
            in FREQUENCIES
        ]
    )

    ax.set_xlabel(
        "Window size (samples)"
    )

    ax.set_ylabel(
        "Dominant frequency (Hz)"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        "Mean |Pearson r|"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "pc_frequency_window_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

def plot_best_window(
    best
):
    fig, ax = plt.subplots(
        figsize=(
            7.0,
            4.6,
        )
    )

    ax.plot(
        best[
            "frequency_hz"
        ],
        best[
            "best_window_size"
        ],
        marker="o",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Dominant frequency (Hz)"
    )

    ax.set_ylabel(
        "Best window size (samples)"
    )

    ax.set_yticks(
        WINDOW_SIZES
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "best_window_vs_frequency.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

def write_configuration():
    cwt_max = float(
        np.max(
            CWT_FREQUENCIES
        )
    )

    cwt_min = float(
        np.min(
            CWT_FREQUENCIES
        )
    )

    frequency_band_warning = (
        max(
            FREQUENCIES
        )
        > cwt_max
    )

    configuration = {
        "purpose": (
            "Feature-level frequency-window analysis using "
            "Power and absolute Pearson correlation with AIC binary labels."
        ),
        "windows": WINDOW_SIZES,
        "feature": FEATURE,
        "frequencies_hz": FREQUENCIES,
        "snr_db": SNR_VALUES,
        "waveform": WAVEFORM,
        "noise_type": NOISE_TYPE,
        "n_repeat": N_REPEAT,
        "base_seed": BASE_SEED,
        "aic_min_side": AIC_MIN_SIDE,
        "aic_eps": _AIC_EPS,
        "label_definition": (
            "Continuous 1 interval bounded by forward and reverse AIC "
            "change points of the noise-free clean waveform."
        ),
        "pc_definition": (
            "absolute Pearson correlation between Power curve "
            "and AIC-derived binary label"
        ),
        "preprocessing": (
            "CWT -> calibrated HOS -> numerical iCWT"
        ),
        "calibration_source": (
            CALIBRATION[
                "source"
            ]
        ),
        "calibration_csv": (
            CALIBRATION.get(
                "csv_path",
                None,
            )
        ),
        "wavelet_name": (
            WAVELET_NAME
        ),
        "cwt_frequency_min_hz": (
            cwt_min
        ),
        "cwt_frequency_max_hz": (
            cwt_max
        ),
        "dominant_frequency_exceeds_cwt_band": (
            frequency_band_warning
        ),
        "experiment_common_file": str(
            Path(
                ec.__file__
            ).resolve()
        ),
    }

    with open(
        OUTPUT_DIR
        / "run_configuration.txt",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            ensure_ascii=False,
        )

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cwt_max = float(
        np.max(
            CWT_FREQUENCIES
        )
    )

    if (
        max(
            FREQUENCIES
        )
        > cwt_max
    ):
        warnings.warn(
            "At least one tested dominant frequency exceeds the maximum "
            f"CWT analysis frequency ({cwt_max:.1f} Hz). Results at those "
            "frequencies may reflect CWT-band truncation as well as the "
            "window effect. Consider extending CWT_FREQUENCIES before "
            "using those points in the manuscript.",
            RuntimeWarning,
        )

    write_configuration()

    print(
        "=" * 78
    )

    print(
        "Power PC vs AIC binary label: frequency-window analysis"
    )

    print(
        "=" * 78
    )

    print(
        f"Windows: {WINDOW_SIZES}"
    )

    print(
        f"Frequencies: {FREQUENCIES}"
    )

    print(
        f"SNR: {SNR_VALUES}"
    )

    print(
        f"Feature: {FEATURE}"
    )

    print(
        f"Repeats / frequency: {N_REPEAT}"
    )

    print(
        f"CWT analysis band: "
        f"{np.min(CWT_FREQUENCIES):.2f} - "
        f"{np.max(CWT_FREQUENCIES):.2f} Hz"
    )

    print(
        f"Calibration: {CALIBRATION['source']}"
    )

    print(
        "=" * 78
    )

    tasks = [
        (
            frequency,
            snr_db,
            repeat,
        )
        for frequency
        in FREQUENCIES
        for snr_db
        in SNR_VALUES
        for repeat
        in range(
            N_REPEAT
        )
    ]

    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):
        nested = Parallel(
            n_jobs=N_JOBS,
            verbose=10,
            max_nbytes="1M",
            mmap_mode="r",
        )(
            delayed(
                run_one_record
            )(
                frequency,
                snr_db,
                repeat,
            )
            for (
                frequency,
                snr_db,
                repeat,
            )
            in tasks
        )

    rows = [
        row
        for record_rows
        in nested
        for row
        in record_rows
    ]

    raw = pd.DataFrame(
        rows
    )

    raw.to_csv(
        OUTPUT_DIR
        / "raw_pc_results.csv",
        index=False,
    )

    summary = (
        summarize_pc(
            raw
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "summary_pc_by_frequency_window.csv",
        index=False,
    )

    best = (
        best_window_table(
            summary
        )
    )

    best.to_csv(
        OUTPUT_DIR
        / "best_window_by_frequency.csv",
        index=False,
    )

    trend = (
        frequency_window_trend(
            best
        )
    )

    trend.to_csv(
        OUTPUT_DIR
        / "best_window_frequency_trend.csv",
        index=False,
    )

    plot_pc_window_curves(
        summary
    )

    plot_pc_heatmap(
        summary
    )

    plot_best_window(
        best
    )

    print()
    print(
        "Best window by dominant frequency:"
    )

    print(
        best.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
                if np.isfinite(
                    value
                )
                else "nan"
            ),
        )
    )

    print()
    print(
        "Frequency vs best-window trend:"
    )

    print(
        trend.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.4f}"
                if np.isfinite(
                    value
                )
                else "nan"
            ),
        )
    )

    print()
    print(
        f"Results written to: "
        f"{OUTPUT_DIR.resolve()}"
    )

if __name__ == "__main__":
    main()
