import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import pywt
from fcmeans import FCM
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from sklearn.preprocessing import MinMaxScaler

FS = 1000.0
DT = 1.0 / FS
EXPECTED_SAMPLES = 450

CWT_WAVELET = "cmor1.5-1.0"
CWT_FREQUENCIES = np.logspace(np.log10(10.0), np.log10(250.0), 200)
SELECTED_SCALE_NUMBER = 100
PREPROCESSING = "icwt"
FEATURE_NAMES = ("M", "P", "SLTA")

WINDOW_SIZE = 21

SHORT_SIZE = 30
LONG_SIZE = 150

C_MIN = 2
C_MAX = 10
FCM_M = 2.0
FCM_MAX_ITER = 100
FCM_ERROR = 1e-6
FCM_RANDOM_STATE = 42

PEAK_HEIGHT = 0.55
PEAK_PROMINENCE = 0.12
PEAK_WIDTH = 2
PEAK_DISTANCE = 3
REGION_THRESHOLD = 0.40
ONSET_RATIO = 0.35
ONSET_CONSECUTIVE = 2
ONSET_LOOKBACK = 100
ONSET_BASELINE_PERCENTILE = 10.0
STABLE_SHIFT = 3
ERROR_LIMIT_MS = 50.0

TRUE_PEAK_SAMPLE = 225
WAVELET_LENGTH = 0.10
ASYMMETRIC_RISE_SCALE = 0.65
ASYMMETRIC_DECAY_SCALE = 1.45

AIC_MIN_SIDE = 2
_AIC_EPS = 1e-12
_EPS = np.finfo(float).eps

def _aic_change_point(signal, min_side=AIC_MIN_SIDE):

    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
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

    var1 = np.maximum(var1, _AIC_EPS)
    var2 = np.maximum(var2, _AIC_EPS)
    aic = n1 * np.log(var1) + n2 * np.log(var2)
    return int(k[np.argmin(aic)])

def aic_reference_arrival(clean_waveform):

    x = np.asarray(clean_waveform, dtype=float).ravel()
    peak_index = int(np.argmax(np.abs(x)))
    leading_branch = x[: peak_index + 1]
    arrival = _aic_change_point(leading_branch, AIC_MIN_SIDE)
    return int(arrival)

def ricker_wavelet(frequency, dt=DT, length=WAVELET_LENGTH):

    t = np.arange(-(length / 2), (length / 2) + dt, dt)
    y = (1 - 2 * np.pi**2 * frequency**2 * t**2) * np.exp(
        -np.pi**2 * frequency**2 * t**2
    )
    return t + length / 2, y

def asymmetric_wavelet(frequency, dt=DT, length=WAVELET_LENGTH):

    t = np.arange(-(length / 2), (length / 2) + dt, dt)
    warped_t = np.where(t < 0, t / ASYMMETRIC_RISE_SCALE, t / ASYMMETRIC_DECAY_SCALE)
    y = (1 - 2 * np.pi**2 * frequency**2 * warped_t**2) * np.exp(
        -np.pi**2 * frequency**2 * warped_t**2
    )
    return t + length / 2, y

def embed_wavelet(waveform, frequency):

    if waveform == "ricker":
        _, wavelet = ricker_wavelet(frequency)
    elif waveform == "asymmetric":
        _, wavelet = asymmetric_wavelet(frequency)
    else:
        raise ValueError("waveform must be 'ricker' or 'asymmetric'.")

    wavelet = np.asarray(wavelet, dtype=float).ravel()
    peak_local = int(np.argmax(np.abs(wavelet)))
    start = TRUE_PEAK_SAMPLE - peak_local
    end = start + wavelet.size

    clean = np.zeros(EXPECTED_SAMPLES, dtype=float)
    dst0, dst1 = max(start, 0), min(end, EXPECTED_SAMPLES)
    src0 = max(-start, 0)
    src1 = src0 + (dst1 - dst0)
    clean[dst0:dst1] = wavelet[src0:src1]

    return clean, aic_reference_arrival(clean)

def window_power(data, start, end):

    x = np.asarray(data, dtype=float).ravel()
    return float(np.mean(x[start:end] ** 2))

def add_noise(signal, snr_db):

    noise = np.random.normal(0, np.std(signal) / (10 ** (snr_db / 20)), len(signal))
    return signal + noise

def load_real_noise_csv(path):

    frame = pd.read_csv(Path(path), header=None, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    numeric = numeric.dropna(axis=1, how="all")

    rows = []
    for _, row in numeric.iterrows():
        values = row.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size >= 64:
            rows.append(values)
    if not rows:
        raise ValueError("No usable real-noise records found.")
    return rows

def extract_real_noise_segment(noise_bank, n_samples, rng):

    row_id = int(rng.integers(0, len(noise_bank)))
    row = np.asarray(noise_bank[row_id], dtype=float).ravel()
    if row.size < n_samples:
        row = np.tile(row, int(np.ceil(n_samples / row.size)))
    start = int(rng.integers(0, row.size - n_samples + 1)) if row.size > n_samples else 0
    segment = row[start : start + n_samples].copy()
    segment -= np.mean(segment)
    return segment, row_id, start

def add_real_noise(signal, real_noise, snr_db):

    x = np.asarray(signal,dtype=float).ravel()

    noise = np.asarray(real_noise,dtype=float).ravel()

    if x.size != noise.size:
        raise ValueError(
            "Signal and real noise must have the same length."
        )

    signal_power = np.mean(x ** 2)
    if signal_power <= _EPS:
        raise ValueError("Signal power is zero.")

    target_noise_power = (signal_power / 10.0 ** (float(snr_db) / 10.0))

    current_noise_power = np.mean(noise ** 2)

    scale = np.sqrt(
        target_noise_power
        / max(current_noise_power, _EPS)
    )
    noisy = x + noise * scale
    return (noisy,float(target_noise_power),float(scale))

def simulate(waveform, frequency, snr_db, noise_type="WGN", noise_bank=None, rng=None):

    rng = np.random.default_rng() if rng is None else rng
    clean, true_arrival = embed_wavelet(waveform, frequency)

    if noise_type == "WGN":
        noisy = add_noise(clean, snr_db)
        noise_source_id = -1
        noise_start_sample = -1
        noise_scale = np.nan
    elif noise_type == "real":
        segment, noise_source_id, noise_start_sample = extract_real_noise_segment(
            noise_bank, EXPECTED_SAMPLES, rng
        )
        noisy, _, noise_scale = add_real_noise(clean, segment, snr_db)
    else:
        raise ValueError("noise_type must be 'WGN' or 'real'.")

    metadata = {
        "true_arrival": int(true_arrival),
        "reference_method": "AIC_prepeak_clean",
        "true_peak_sample": int(np.argmax(np.abs(clean))),
        "target_snr_db": float(snr_db),
        "snr_definition": "full_trace_mean_square",
        "noise_source_id": int(noise_source_id),
        "noise_start_sample": int(noise_start_sample),
        "noise_scale": float(noise_scale) if np.isfinite(noise_scale) else np.nan,
    }
    return noisy, clean, metadata

def cwt_morlet_pywt(signal_data, dt=DT, freqs=CWT_FREQUENCIES, wavelet_name=CWT_WAVELET):

    wavelet = pywt.ContinuousWavelet(wavelet_name)
    freqs = np.asarray(freqs, dtype=float)
    scales = pywt.frequency2scale(wavelet, freqs * dt)
    pad = int(np.ceil(8 * np.max(scales)))
    padded = np.pad(np.asarray(signal_data, dtype=float).ravel(), pad, mode="reflect")
    coeffs, actual_freqs = pywt.cwt(padded, scales, wavelet, sampling_period=dt, method="fft")
    return coeffs[:, pad : pad + len(signal_data)], actual_freqs

def get_analysis_scales(freqs=CWT_FREQUENCIES, dt=DT, wavelet_name=CWT_WAVELET):

    wavelet = pywt.ContinuousWavelet(wavelet_name)
    return np.asarray(pywt.frequency2scale(wavelet, np.asarray(freqs) * dt), dtype=float)

def inverse_cwt(coefficients, scales=None):

    coeffs = np.asarray(coefficients)
    scales = get_analysis_scales() if scales is None else np.asarray(scales, dtype=float)
    order = np.argsort(scales)
    scales_sorted = scales[order]
    coeffs_sorted = coeffs[order, :]
    integrand = np.real(coeffs_sorted) / (scales_sorted[:, None] ** 1.5)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    y = np.asarray(trapz(integrand, x=scales_sorted, axis=0), dtype=float).ravel()
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return y

HOS_MODE = "calibrated"          # "calibrated", "original", "none"
HOS_CONFIDENCE_LEVEL = 0.90
HOS_CALIBRATION_QUANTILE = 0.90
HOS_CALIBRATION_FREQUENCIES = CWT_FREQUENCIES.copy()
HOS_CALIBRATED_THRESHOLDS = np.array([
    0.985559751403, 0.981066136847, 1.007165362464, 1.020613365562,
    1.025761928209, 0.983904260006, 0.970069560153, 0.951828146626,
    0.931098797342, 0.917536316710, 0.904681641984, 0.910231849992,
    0.898157645958, 0.908196531813, 0.895597720425, 0.890799755765,
    0.887029942897, 0.886179451660, 0.878872509454, 0.877368079695,
    0.894568681564, 0.886381736207, 0.885805253571, 0.837958953904,
    0.827582152933, 0.791743634325, 0.803588613220, 0.808595137436,
    0.802027548652, 0.803466365504, 0.786968814184, 0.777339258417,
    0.753365374193, 0.732721252333, 0.728574096534, 0.718993428725,
    0.713473845636, 0.697978692638, 0.702255254658, 0.688617120036,
    0.656253545121, 0.646919629229, 0.666662451698, 0.662810974276,
    0.642962517199, 0.637021849875, 0.619767937281, 0.604908135575,
    0.604773551987, 0.594117458359, 0.599101516848, 0.610531200282,
    0.594947785670, 0.573594644044, 0.549723593654, 0.540989669737,
    0.535741781868, 0.515595039424, 0.514060185841, 0.505582147746,
    0.498307462237, 0.481354463270, 0.473126558162, 0.479738039821,
    0.481031920834, 0.491161987215, 0.486129417193, 0.465701602404,
    0.464027267764, 0.444129565099, 0.430692230987, 0.427481220438,
    0.429508356616, 0.412887429405, 0.403580570332, 0.395035060730,
    0.374217668288, 0.374244672608, 0.377125484699, 0.378477887892,
    0.393065831578, 0.415091784068, 0.414943338437, 0.399098879654,
    0.380248250994, 0.360897281176, 0.346311855473, 0.333740447163,
    0.321506370508, 0.323038048884, 0.321678881314, 0.326092926619,
    0.316044497403, 0.307465996077, 0.335716872357, 0.350161599625,
    0.317992925033, 0.390417048840, 0.412410481914, 0.317968206424,
], dtype=float)

def hos_preprocess_cwt(coeffs, freqs, confidence_level=0.9):

    n_scales, n_times = coeffs.shape
    filtered_coeffs = coeffs.copy()

    kurtosis_values = np.zeros(n_scales)
    for i in range(n_scales):
        kurtosis_values[i] = kurtosis(np.abs(coeffs[i, :]), fisher=True)

    bias = -6 / n_times
    variance = 24 / n_times
    threshold = np.sqrt(variance / (1 - confidence_level))

    kept_idx = np.where(np.abs(kurtosis_values - bias) > threshold)[0]

    scales_to_remove = np.abs(kurtosis_values - bias) <= threshold
    filtered_coeffs[scales_to_remove, :] = 0

    return filtered_coeffs, kept_idx

def cwt_hos_icwt(signal, thresholds=None, hos_mode=HOS_MODE):

    coeffs, freqs = cwt_morlet_pywt(signal)

    filtered, kept_idx = hos_preprocess_cwt(coeffs, freqs)
    enhanced = inverse_cwt(filtered)
    return enhanced, kept_idx, freqs

def cwt_hos_sum(signal, thresholds=None, hos_mode=HOS_MODE):

    coeffs, freqs = cwt_morlet_pywt(signal)

    filtered, kept_idx = hos_preprocess_cwt(coeffs, freqs)
    enhanced = np.abs(filtered[kept_idx]).sum(axis=0) if kept_idx.size else np.zeros(len(signal))
    return enhanced, kept_idx, freqs

def _window_bounds(n, window_size):
    half = int(window_size) // 2
    idx = np.arange(n)
    starts = np.maximum(0, idx - half)
    ends = np.minimum(n, idx + half + 1)
    return starts, ends

def get_amplitude(data, window_size=WINDOW_SIZE):

    x = np.abs(np.asarray(data, dtype=float).ravel())

    starts, ends = _window_bounds(x.size, window_size)
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[ends] - c[starts]) / (ends - starts)

def get_energy(data, window_size=WINDOW_SIZE):

    x2 = np.asarray(data, dtype=float).ravel() ** 2
    starts, ends = _window_bounds(x2.size, window_size)
    c = np.concatenate(([0.0], np.cumsum(x2)))
    return c[ends] - c[starts]

def get_SLTA(data, Nsta=SHORT_SIZE, Nlta=LONG_SIZE):

    x2 = np.abs(np.asarray(data, dtype=float).ravel()) ** 2
    n = x2.size

    c = np.concatenate(([0.0], np.cumsum(x2)))

    starts_sta, ends_sta = _window_bounds(n, Nsta)
    starts_lta, ends_lta = _window_bounds(n, Nlta)

    sum_sta = c[ends_sta] - c[starts_sta]
    sum_lta = c[ends_lta] - c[starts_lta]

    lens_sta = ends_sta - starts_sta
    lens_lta = ends_lta - starts_lta

    sta = sum_sta / lens_sta
    lta = sum_lta / lens_lta

    return np.where(lta > 0, sta / lta, 0.0)

def get_std(data, window_size=WINDOW_SIZE):

    x = np.asarray(data, dtype=float).ravel()
    starts, ends = _window_bounds(x.size, window_size)
    out = np.zeros(x.size)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i] = np.std(x[s:e], ddof=0)
    return out

def get_kurtosis(data, window_size=WINDOW_SIZE):

    x = np.asarray(data, dtype=float).ravel()
    starts, ends = _window_bounds(x.size, window_size)
    out = np.zeros(x.size)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i] = kurtosis(x[s:e], fisher=False, bias=True) if e > s + 1 else 0.0
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def get_skewness(data, window_size=WINDOW_SIZE):

    x = np.asarray(data, dtype=float).ravel()
    starts, ends = _window_bounds(x.size, window_size)
    out = np.zeros(x.size)
    for i, (s, e) in enumerate(zip(starts, ends)):
        w = x[s:e]
        sigma = np.std(w, ddof=0)
        out[i] = np.mean(((w - np.mean(w)) / sigma) ** 3) if sigma > _EPS else 0.0
    return out

def build_feature_matrix(enhanced, feature_names=FEATURE_NAMES, window_size=WINDOW_SIZE):

    x = np.asarray(enhanced, dtype=float).ravel()
    feature_map = {
        "M": get_amplitude(x, window_size),
        "P": get_energy(x, window_size),
        "SLTA": get_SLTA(x, SHORT_SIZE, LONG_SIZE),
        "STD": get_std(x, window_size),
        "K": get_kurtosis(x, window_size),
        "S": get_skewness(x, window_size),
    }
    features = np.column_stack([feature_map[name.upper()] for name in feature_names]).astype(float)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if np.all(np.ptp(features, axis=0) <= _EPS):
        return None
    return MinMaxScaler(copy=False).fit_transform(features)

def extract_features(signal, feature_names=FEATURE_NAMES, preprocessing=PREPROCESSING, thresholds=None):

    if preprocessing == "icwt":
        enhanced, kept_idx, _ = cwt_hos_icwt(signal, thresholds=thresholds)
    elif preprocessing == "sum":
        enhanced, kept_idx, _ = cwt_hos_sum(signal, thresholds=thresholds)
    else:
        raise ValueError("preprocessing must be 'icwt' or 'sum'.")

    if kept_idx.size == 0:
        return None
    return build_feature_matrix(enhanced, feature_names)

def _peak_region_bounds(curve, peak):
    left = right = int(peak)
    while left > 0 and curve[left - 1] >= REGION_THRESHOLD:
        left -= 1
    while right + 1 < curve.size and curve[right + 1] >= REGION_THRESHOLD:
        right += 1
    return left, right

def merge_region_peaks(curve, peaks, prominences):

    curve = np.asarray(curve, dtype=float).ravel()
    peaks = np.asarray(peaks, dtype=int).ravel()
    prominences = np.asarray(prominences, dtype=float).ravel()
    if peaks.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    order = np.argsort(peaks)
    peaks, prominences = peaks[order], prominences[order]
    groups, group, current_right = [], [], -1

    for i, peak in enumerate(peaks):
        left, right = _peak_region_bounds(curve, peak)
        if group and left > current_right:
            groups.append(group)
            group = []
        group.append(i)
        current_right = max(current_right, right)
    if group:
        groups.append(group)

    merged_peaks, merged_proms = [], []
    for g in groups:
        best = max(g, key=lambda i: (prominences[i], curve[peaks[i]], -peaks[i]))
        merged_peaks.append(int(peaks[best]))
        merged_proms.append(float(prominences[best]))

    return np.asarray(merged_peaks, dtype=int), np.asarray(merged_proms, dtype=float)

def backward_pick_relative(curve, peak):

    curve = np.asarray(curve, dtype=float).ravel()
    peak = int(peak)
    search_start = max(0, peak - int(ONSET_LOOKBACK))
    local_curve = curve[search_start : peak + 1]
    baseline = float(np.percentile(local_curve, ONSET_BASELINE_PERCENTILE))
    peak_value = float(curve[peak])
    onset_level = baseline + ONSET_RATIO * (peak_value - baseline)

    count = 0
    for index in range(peak - 1, search_start - 1, -1):
        if curve[index] < onset_level:
            count += 1
            if count >= ONSET_CONSECUTIVE:
                return int(index + ONSET_CONSECUTIVE), float(onset_level), baseline
        else:
            count = 0
    return None, float(onset_level), baseline

def _empty_adaptive_result(cluster_num=C_MIN, stop_reason="not_started"):
    return {
        "arrival": None,
        "cluster_num": int(cluster_num),
        "stable_count": 0,
        "valid_peak_count": 0,
        "valid_peaks": [],
        "peak": None,
        "peak_membership": np.nan,
        "onset_level": np.nan,
        "onset_baseline": np.nan,
        "stop_reason": stop_reason,
        "diagnostic_message": "",
        "success": False,
    }

def adaptive_fcm(features):

    if features is None:
        return _empty_adaptive_result(stop_reason="empty_feature")

    previous_peak, stable_count = None, 0
    result = _empty_adaptive_result()
    fitted = 0
    last_error = ""

    for cluster_num in range(C_MIN, C_MAX + 1):
        result["cluster_num"] = int(cluster_num)
        try:
            model = FCM(
                n_clusters=cluster_num,
                max_iter=FCM_MAX_ITER,
                m=FCM_M,
                error=FCM_ERROR,
                random_state=FCM_RANDOM_STATE,
            )
            model.fit(features)
            centers = np.asarray(model.centers)
            membership = np.asarray(model.u)
        except Exception as error:
            last_error = str(error)
            previous_peak, stable_count = None, 0
            continue

        fitted += 1
        signal_cluster = int(np.argmax(centers.mean(axis=1)))
        signal_membership = membership[:, signal_cluster]

        peaks, props = find_peaks(
            signal_membership,
            height=PEAK_HEIGHT,
            prominence=PEAK_PROMINENCE,
            distance=PEAK_DISTANCE,
            width=PEAK_WIDTH,
        )
        valid_peaks, _ = merge_region_peaks(signal_membership, peaks, props["prominences"])
        peak = int(valid_peaks[0]) if valid_peaks.size else None

        if peak is None:
            previous_peak, stable_count = None, 0
        elif previous_peak is not None and abs(peak - previous_peak) <= STABLE_SHIFT:
            stable_count += 1
        else:
            stable_count = 1
        previous_peak = peak

        result.update(
            {
                "stable_count": int(stable_count),
                "valid_peak_count": int(valid_peaks.size),
                "valid_peaks": valid_peaks.astype(int).tolist(),
                "peak": peak,
                "peak_membership": float(signal_membership[peak]) if peak is not None else np.nan,
                "stop_reason": "no_valid_peak" if valid_peaks.size == 0 else "multiple_valid_peaks",
                "success": False,
            }
        )

        if valid_peaks.size != 1:
            continue

        arrival, onset_level, baseline = backward_pick_relative(signal_membership, peak)
        result.update({"arrival": arrival, "onset_level": onset_level, "onset_baseline": baseline})

        if arrival is None:
            result["stop_reason"] = "unique_peak_no_onset_crossing"
            return result

        result["stop_reason"] = "unique_valid_peak"
        result["success"] = True
        return result

    if fitted == 0:
        result["stop_reason"] = "fcm_failed_all_clusters"
        result["diagnostic_message"] = last_error
    elif result["valid_peak_count"] == 0:
        result["stop_reason"] = "no_valid_peak_at_cmax"
    else:
        result["stop_reason"] = "multiple_valid_peaks_at_cmax"
    return result

def normalize_pick_result(result):

    arrival = result.get("arrival")
    peak = result.get("peak")
    success = bool(result.get("success", False))
    return {
        "predicted_arrival": int(arrival) if arrival is not None and np.isfinite(arrival) else np.nan,
        "cluster_num": result.get("cluster_num", np.nan),
        "stable_count": int(result.get("stable_count", 0)),
        "valid_peak_count": int(result.get("valid_peak_count", 0)),
        "peak_position": int(peak) if peak is not None and np.isfinite(peak) else np.nan,
        "peak_membership": result.get("peak_membership", np.nan),
        "onset_level": result.get("onset_level", np.nan),
        "onset_baseline": result.get("onset_baseline", np.nan),
        "stop_reason": result.get("stop_reason", "unreliable"),
        "success": success,
        "status": "Reliable" if success else "Unreliable",
        "message": result.get("diagnostic_message", ""),
    }

def pick_signal(signal, feature_names=FEATURE_NAMES, preprocessing=PREPROCESSING, thresholds=None):

    try:
        features = extract_features(signal, feature_names, preprocessing, thresholds)
        return normalize_pick_result(adaptive_fcm(features))
    except Exception as error:
        return {
            "predicted_arrival": np.nan,
            "cluster_num": np.nan,
            "stable_count": 0,
            "valid_peak_count": 0,
            "peak_position": np.nan,
            "peak_membership": np.nan,
            "onset_level": np.nan,
            "onset_baseline": np.nan,
            "stop_reason": "exception",
            "success": False,
            "status": "Failed",
            "message": str(error),
        }
