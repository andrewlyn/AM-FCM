import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

import experiment_common as ec

WAVEFORM = "ricker"
NOISE_TYPE = "WGN"

WINDOW_SIZES = [7, 9, 11, 13, 15, 17, 21]

FREQUENCIES = [100, 150, 200, 250, 300]

SNR_VALUES = [-10, -8, -6, -4, -2, 0, 2, 4, 6]

N_REPEAT = 200
N_JOBS = 32
BASE_SEED = 20260819
SAVE_RAW = False

FEATURE_NAMES = ("P", "M", "Std", "SLTA", "K", "S")
OUTPUT_DIR = Path("results_window_feature_relevance")

def build_binary_labels(clean):

    x = np.asarray(clean, dtype=float).ravel()
    onset = ec.aic_reference_arrival(x)
    reverse_onset = ec.aic_reference_arrival(x[::-1])
    end = x.size - 1 - reverse_onset
    peak = int(np.argmax(np.abs(x)))

    if not (onset <= peak <= end):
        raise RuntimeError(f"Invalid AIC label: onset={onset}, peak={peak}, end={end}")

    labels = np.zeros(x.size, dtype=np.int8)
    labels[onset:end + 1] = 1
    return labels, int(onset), int(end)

def calculate_features(enhanced, window_size):

    x = np.asarray(enhanced, dtype=float).ravel()

    feature_map = {
        "P": ec.get_energy(x, window_size),
        "M": ec.get_amplitude(x, window_size),
        "Std": ec.get_std(x, window_size),
        "SLTA": ec.get_SLTA(x, ec.SHORT_SIZE, ec.LONG_SIZE),
        "K": ec.get_kurtosis(x, window_size),
        "S": ec.get_skewness(x, window_size),
    }

    features = np.column_stack([feature_map[name] for name in FEATURE_NAMES]).astype(float)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def safe_abs_corr(func, feature, labels):
    if np.std(feature) <= np.finfo(float).eps or np.std(labels) <= np.finfo(float).eps:
        return 0.0
    value = func(feature, labels)[0]
    return float(abs(value)) if np.isfinite(value) else 0.0

def calculate_scores(features, labels, seed):
    scaled = MinMaxScaler().fit_transform(features)

    pearson = np.array([
        safe_abs_corr(pearsonr, scaled[:, i], labels)
        for i in range(scaled.shape[1])
    ])

    spearman = np.array([
        safe_abs_corr(spearmanr, scaled[:, i], labels)
        for i in range(scaled.shape[1])
    ])

    mi = mutual_info_classif(
        scaled,
        labels,
        discrete_features=False,
        n_neighbors=3,
        random_state=seed,
    )
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    return pearson, spearman, mi

def evaluate_one(frequency_index, frequency, snr_db, repeat_id, labels):

    seed_seq = np.random.SeedSequence([BASE_SEED, frequency_index, repeat_id])
    seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(seed)

    noisy, _, metadata = ec.simulate(
        WAVEFORM, frequency, snr_db, noise_type=NOISE_TYPE, rng=rng
    )
    enhanced, kept_idx, _ = ec.cwt_hos_icwt(noisy)

    rows = []
    for window_size in WINDOW_SIZES:
        features = calculate_features(enhanced, window_size)
        pearson, spearman, mi = calculate_scores(
            features, labels, seed + int(window_size)
        )

        for i, feature in enumerate(FEATURE_NAMES):
            rows.append({
                "window_size": int(window_size),
                "frequency_hz": int(frequency),
                "snr_db": int(snr_db),
                "repeat": int(repeat_id),
                "feature": feature,
                "pearson_abs": float(pearson[i]),
                "spearman_abs": float(spearman[i]),
                "mutual_information": float(mi[i]),
                "retained_scale_count": int(kept_idx.size),
                "snr_definition": metadata.get("snr_definition", ""),
            })

    return rows

def summarize_conditions(raw):

    summary = raw.groupby(
        ["window_size", "frequency_hz", "snr_db", "feature"],
        as_index=False,
        observed=True,
    ).agg(
        n_records=("repeat", "count"),
        pearson_mean=("pearson_abs", "mean"),
        pearson_std=("pearson_abs", "std"),
        spearman_mean=("spearman_abs", "mean"),
        spearman_std=("spearman_abs", "std"),
        mi_mean=("mutual_information", "mean"),
        mi_std=("mutual_information", "std"),
        mean_retained_scale_count=("retained_scale_count", "mean"),
    )

    summary["pearson_rank"] = summary.groupby(
        ["window_size", "frequency_hz", "snr_db"], observed=True
    )["pearson_mean"].rank(ascending=False, method="average")

    summary["spearman_rank"] = summary.groupby(
        ["window_size", "frequency_hz", "snr_db"], observed=True
    )["spearman_mean"].rank(ascending=False, method="average")

    summary["mi_rank"] = summary.groupby(
        ["window_size", "frequency_hz", "snr_db"], observed=True
    )["mi_mean"].rank(ascending=False, method="average")

    summary["pearson_top3"] = summary["pearson_rank"] <= 3
    summary["spearman_top3"] = summary["spearman_rank"] <= 3
    summary["mi_top3"] = summary["mi_rank"] <= 3
    return summary

def summarize_top3(summary):

    top3 = summary.groupby(
        ["window_size", "feature"], as_index=False, observed=True
    ).agg(
        condition_count=("snr_db", "count"),
        pearson_top3_count=("pearson_top3", "sum"),
        spearman_top3_count=("spearman_top3", "sum"),
        mi_top3_count=("mi_top3", "sum"),
        pearson_mean_rank=("pearson_rank", "mean"),
        spearman_mean_rank=("spearman_rank", "mean"),
        mi_mean_rank=("mi_rank", "mean"),
    )

    top3["pearson_top3_pct"] = 100.0 * top3["pearson_top3_count"] / top3["condition_count"]
    top3["spearman_top3_pct"] = 100.0 * top3["spearman_top3_count"] / top3["condition_count"]
    top3["mi_top3_pct"] = 100.0 * top3["mi_top3_count"] / top3["condition_count"]
    top3["mean_top3_pct"] = top3[
        ["pearson_top3_pct", "spearman_top3_pct", "mi_top3_pct"]
    ].mean(axis=1)
    return top3

def make_top3_wide(top3, selected_features=("P", "M", "Std")):
    part = top3[top3["feature"].isin(selected_features)].copy()
    wide = part.pivot(
        index="window_size",
        columns="feature",
        values=["pearson_top3_pct", "spearman_top3_pct", "mi_top3_pct", "mean_top3_pct"],
    )
    wide.columns = [f"{feature}_{metric}" for metric, feature in wide.columns]
    return wide.reset_index()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_cache = {}
    for frequency in FREQUENCIES:
        clean, _ = ec.embed_wavelet(WAVEFORM, frequency)
        labels, onset, end = build_binary_labels(clean)
        label_cache[frequency] = (labels, onset, end)

    print("=" * 76)
    print("Window-feature relevance and Top-3 stability")
    print("=" * 76)
    print(f"Windows: {WINDOW_SIZES}")
    print(f"Frequencies: {FREQUENCIES} Hz")
    print(f"SNR: {SNR_VALUES} dB")
    print(f"Repeats: {N_REPEAT}")
    print(f"Features: {FEATURE_NAMES}")
    print(f"STA/LTA fixed at: {ec.SHORT_SIZE}/{ec.LONG_SIZE}")
    print(f"Parallel jobs: {N_JOBS}")
    print("-" * 76)

    for frequency in FREQUENCIES:
        _, onset, end = label_cache[frequency]
        print(f"{frequency:3d} Hz: AIC label {onset}-{end}, width={end - onset + 1}")

    tasks = []
    for frequency_index, frequency in enumerate(FREQUENCIES):
        labels, _, _ = label_cache[frequency]
        for snr_db in SNR_VALUES:
            for repeat_id in range(N_REPEAT):
                tasks.append(
                    (frequency_index, frequency, snr_db, repeat_id, labels)
                )

    print(f"\nRealizations: {len(tasks)}")
    print(f"Final feature-score rows: {len(tasks) * len(WINDOW_SIZES) * len(FEATURE_NAMES)}")
    print("Starting computation...")

    with parallel_backend("loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(
            n_jobs=N_JOBS,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
            verbose=10,
        )(
            delayed(evaluate_one)(*task)
            for task in tasks
        )

    raw = pd.DataFrame([row for block in results for row in block])
    summary = summarize_conditions(raw)
    top3 = summarize_top3(summary)
    top3_wide = make_top3_wide(top3)

    if SAVE_RAW:
        raw.to_csv(
            OUTPUT_DIR / "window_feature_relevance_raw.csv",
            index=False, encoding="utf-8-sig"
        )

    summary.to_csv(
        OUTPUT_DIR / "window_feature_mean_by_condition.csv",
        index=False, encoding="utf-8-sig"
    )
    top3.to_csv(
        OUTPUT_DIR / "top3_occurrence_by_window_all_features.csv",
        index=False, encoding="utf-8-sig"
    )
    top3_wide.to_csv(
        OUTPUT_DIR / "top3_occurrence_by_window_P_M_Std.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\nP/M/Std Top-3 occurrence (%)")
    print("-" * 76)
    display_cols = [
        "window_size", "feature",
        "pearson_top3_pct", "spearman_top3_pct",
        "mi_top3_pct", "mean_top3_pct",
    ]
    print(
        top3[top3["feature"].isin(["P", "M", "Std"])][display_cols]
        .round(2)
        .to_string(index=False)
    )
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
