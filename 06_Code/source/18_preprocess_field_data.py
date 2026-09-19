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
from scipy.signal import butter, sosfiltfilt
from joblib import Parallel, delayed, parallel_config

DATA_ROOT = Path("./orgin_data")
OUTPUT_ROOT = Path("./pre_data")

SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"

OUTPUT_SIGNAL_DIR = OUTPUT_ROOT / "signal"
OUTPUT_LABEL_DIR = OUTPUT_ROOT / "labels"
OUTPUT_FIG_DIR = OUTPUT_ROOT / "fig"

FS = 1000.0
HIGH_PASS_FREQ = 100.0
FILTER_ORDER = 4

CROP_LENGTH = 1000
RANDOM_SEED = 42
N_JOBS = 6

ZOOM_HALF_WIDTH = 100

def read_signal(path):
    data = np.loadtxt(path)
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(float)

def read_label(path):

    data = np.loadtxt(path)
    arrival = int(round(float(np.asarray(data).reshape(-1)[0])))
    return arrival

def highpass(signal):
    sos = butter(
        FILTER_ORDER,
        HIGH_PASS_FREQ,
        btype="highpass",
        fs=FS,
        output="sos"
    )
    return sosfiltfilt(sos, signal)

def crop_by_arrival(signal, arrival, rng):

    n = len(signal)

    if n < CROP_LENGTH:
        raise ValueError(f"Signal has fewer than 1000 samples: signal length={n}")

    arrival = int(np.clip(arrival, 0, n - 1))

    start_min = max(0, arrival - CROP_LENGTH + 1)
    start_max = min(arrival, n - CROP_LENGTH)

    if start_min > start_max:
        raise ValueError(
            f"Cannot crop: length={n}, arrival={arrival}, start range={start_min}~{start_max}"
        )

    start = int(rng.integers(start_min, start_max + 1))
    end = start + CROP_LENGTH
    cropped = signal[start:end]
    arrival_local = arrival - start

    return cropped, arrival_local, start, arrival

def plot_cropped_signal(signal, arrival_local, file_stem):

    t = np.arange(len(signal)) / FS
    arrival_t = arrival_local / FS

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1, ax2 = axes

    ax1.plot(t, signal, linewidth=0.8, label="100 Hz high-pass")
    ax1.axvline(arrival_t, color="red", linestyle="-", label="Manual arrival")
    ax1.set_title(f"id={file_stem}, cropped signal, arrival={arrival_local}")
    ax1.set_ylabel("Amplitude")
    ax1.grid(linestyle="--", alpha=0.3)
    ax1.legend()

    left = max(0, arrival_local - ZOOM_HALF_WIDTH)
    right = min(len(signal), arrival_local + ZOOM_HALF_WIDTH)

    ax2.plot(t[left:right], signal[left:right], linewidth=1.0)
    ax2.axvline(arrival_t, color="red", linestyle="-")
    ax2.set_title("arrival zoom region")
    ax2.set_xlabel("Time / s")
    ax2.set_ylabel("Amplitude")
    ax2.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_FIG_DIR / f"{file_stem}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

def process_signal(path):
    try:
        label_path = LABEL_DIR / path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path.name}")

        raw = read_signal(path)
        arrival_original = read_label(label_path)

        filtered = highpass(raw)

        try:
            file_seed = int(path.stem)
        except:
            file_seed = abs(hash(path.stem)) % 1000000

        rng = np.random.default_rng(RANDOM_SEED + file_seed)

        cropped, arrival_local, start, arrival_used = crop_by_arrival(
            filtered, arrival_original, rng
        )

        np.savetxt(
            OUTPUT_SIGNAL_DIR / path.name,
            cropped,
            fmt="%.10e"
        )

        np.savetxt(
            OUTPUT_LABEL_DIR / path.name,
            [arrival_local],
            fmt="%d"
        )

        plot_cropped_signal(cropped, arrival_local, path.stem)

        return {
            "success": True,
            "file": path.name,
            "arrival_original": arrival_original,
            "arrival_used": arrival_used,
            "crop_start": start,
            "arrival_local": arrival_local
        }

    except Exception as e:
        print(f"{path.name}: {e}")
        return {
            "success": False,
            "file": path.name,
            "arrival_original": np.nan,
            "arrival_used": np.nan,
            "crop_start": np.nan,
            "arrival_local": np.nan
        }

def main():
    OUTPUT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        SIGNAL_DIR.glob("*.txt"),
        key=lambda x: int(x.stem)
    )

    print("signal records:", len(files))

    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_signal)(f) for f in files
        )

    success = sum(r["success"] for r in results)

    print("finished:", success, "/", len(results))

    df = pd.DataFrame(results)
    df.to_csv(
        OUTPUT_ROOT / "crop_info.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if success > 0:
        valid = df[df["success"]]
        print("arrival_local range:",
              int(valid["arrival_local"].min()),
              "~",
              int(valid["arrival_local"].max()))
        print("arrival_local mean:",
              f"{valid['arrival_local'].mean():.1f}")

    print("output:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
