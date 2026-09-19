from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import seisbench
import seisbench.models as sbm

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from experiment_common import (
    DT,
    EXPECTED_SAMPLES,
    load_real_noise_csv,
    simulate,
)

FREQUENCIES = [100, 150, 200, 250, 300]
SNR_VALUES = [-10, -8, -6, 2]

TEST_REPEATS = 2000

TRAIN_REPEATS = 500
VAL_REPEATS = 100

BASE_SEED = 20260911
NOISE_TYPE = "WGN"

P_LABEL_SIGMA_MS = 3.0
DETECTION_LENGTH_MS = 100.0

PHASENET_BATCH_SIZE = 64
PHASENET_LR = 1e-3
PHASENET_EPOCHS = 50
PHASENET_P_WEIGHT = 8.0

EQT_BATCH_SIZE = 32
EQT_LR = 5e-4
EQT_EPOCHS = 50
EQT_DETECTION_LOSS_WEIGHT = 0.2
EQT_P_LOSS_WEIGHT = 1.0
EQT_DETECTION_POS_WEIGHT = 3.0
EQT_P_POS_WEIGHT = 20.0

EARLY_STOP_PATIENCE = 8
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

PROBABILITY_LIMITS_MS = (2.0, 5.0, 10.0)
OUTPUT_DIR = Path(__file__).resolve().parent / "results_phasenet_eqtransformer"

AIC_MIN_SIDE = 2
AIC_EPS = 1e-12

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(name: str) -> torch.device:

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def _aic_change_point(signal: np.ndarray, min_side: int = AIC_MIN_SIDE) -> int:

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
    return int(k[np.argmin(aic)])

def clean_aic_reference(clean: np.ndarray) -> int:

    x = np.asarray(clean, dtype=float).ravel()
    peak = int(np.argmax(np.abs(x)))
    return int(_aic_change_point(x[: peak + 1]))

def normalize_trace(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=np.float32).ravel()
    x = x - np.mean(x)
    std = float(np.std(x))
    if std > 1e-8:
        x = x / std
    return x.astype(np.float32, copy=False)

def gaussian_target(n: int, center: int, sigma_samples: float) -> np.ndarray:

    idx = np.arange(n, dtype=np.float32)
    y = np.exp(-0.5 * ((idx - float(center)) / float(sigma_samples)) ** 2)
    return y.astype(np.float32)

def detection_target(n: int, arrival: int, length_samples: int) -> np.ndarray:

    y = np.zeros(n, dtype=np.float32)
    start = max(0, int(arrival))
    end = min(n, start + int(length_samples))
    y[start:end] = 1.0
    return y

def generate_split(
    split_name: str,
    frequencies: list[int],
    snrs: list[float],
    repeats: int,
    base_seed: int,
    noise_bank,
) -> dict:

    split_offsets = {"train": 0, "val": 100_000_000, "test": 200_000_000}
    if split_name not in split_offsets:
        raise ValueError("split_name must be train / val / test.")

    seed_rng = np.random.default_rng(base_seed + split_offsets[split_name])
    total = len(frequencies) * len(snrs) * repeats
    seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=total, dtype=np.uint32)

    x_all = np.empty((total, 1, EXPECTED_SAMPLES), dtype=np.float32)
    p_all = np.empty((total, EXPECTED_SAMPLES), dtype=np.float32)
    d_all = np.empty((total, EXPECTED_SAMPLES), dtype=np.float32)
    meta_rows = []

    sigma_samples = P_LABEL_SIGMA_MS / (DT * 1000.0)
    det_samples = max(1, int(round(DETECTION_LENGTH_MS / (DT * 1000.0))))

    idx = 0
    for frequency in frequencies:
        for snr_db in snrs:
            for repeat_id in range(repeats):
                seed = int(seeds[idx])
                rng = np.random.default_rng(seed)
                np.random.seed(seed)

                noise_type = NOISE_TYPE if noise_bank is None else "real"
                noisy, clean, metadata = simulate(
                    waveform="ricker",
                    frequency=frequency,
                    snr_db=snr_db,
                    noise_type=noise_type,
                    noise_bank=noise_bank,
                    rng=rng,
                )
                noisy = np.asarray(noisy, dtype=float).ravel()
                clean = np.asarray(clean, dtype=float).ravel()

                if noisy.size != EXPECTED_SAMPLES:
                    raise RuntimeError(
                        f"Record length {noisy.size}  does not match EXPECTED_SAMPLES={EXPECTED_SAMPLES}  do not match."
                    )

                reference = clean_aic_reference(clean)
                common_reference = int(metadata["true_arrival"])
                if reference != common_reference:
                    raise RuntimeError(
                        f"{split_name}: clean-AIC={reference}, "
                        f"simulate reference={common_reference}; reference arrival definitions differ."
                    )

                x_all[idx, 0] = normalize_trace(noisy)
                p_all[idx] = gaussian_target(EXPECTED_SAMPLES, reference, sigma_samples)
                d_all[idx] = detection_target(EXPECTED_SAMPLES, reference, det_samples)

                meta_rows.append(
                    {
                        "split": split_name,
                        "frequency_hz": int(frequency),
                        "snr_db": float(snr_db),
                        "repeat_id": int(repeat_id),
                        "seed": seed,
                        "reference_arrival_sample": int(reference),
                        "reference_arrival_ms": float(reference * DT * 1000.0),
                        "noise_type": noise_type,
                        "noise_source_id": int(metadata["noise_source_id"]),
                        "noise_start_sample": int(metadata["noise_start_sample"]),
                    }
                )
                idx += 1

    return {
        "x": x_all,
        "p": p_all,
        "detection": d_all,
        "meta": pd.DataFrame(meta_rows),
    }

def make_loader(split: dict, batch_size: int, shuffle: bool) -> DataLoader:

    dataset = TensorDataset(
        torch.from_numpy(split["x"]),
        torch.from_numpy(split["p"]),
        torch.from_numpy(split["detection"]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

def build_phasenet() -> torch.nn.Module:

    return sbm.VariableLengthPhaseNet(
        in_samples=EXPECTED_SAMPLES,
        in_channels=1,
        classes=2,
        phases="PN",
        sampling_rate=1.0 / DT,
        norm="std",
        norm_axis=(-1,),
        output_activation="softmax",
    )

def build_eqtransformer() -> torch.nn.Module:

    return sbm.EQTransformer(
        in_channels=1,
        in_samples=EXPECTED_SAMPLES,
        classes=1,
        phases="P",
        lstm_blocks=3,
        drop_rate=0.1,
        original_compatible=False,
        sampling_rate=1.0 / DT,
        norm="std",
    )

def phasenet_loss(model: torch.nn.Module, x: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:

    logits = model(x, logits=True)            # [B, 2, T]
    target = torch.stack((p_target, 1.0 - p_target), dim=1)
    log_prob = F.log_softmax(logits, dim=1)

    ce = -(target * log_prob).sum(dim=1)      # [B, T]
    time_weight = 1.0 + (PHASENET_P_WEIGHT - 1.0) * p_target
    return (ce * time_weight).mean()

def eqtransformer_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    p_target: torch.Tensor,
    d_target: torch.Tensor,
) -> torch.Tensor:

    outputs = model(x, logits=True)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError("Unexpected EQTransformer output format; detection and P branches are required.")

    d_logits, p_logits = outputs[0], outputs[1]
    det_pos = torch.tensor(EQT_DETECTION_POS_WEIGHT, device=x.device)
    p_pos = torch.tensor(EQT_P_POS_WEIGHT, device=x.device)

    loss_det = F.binary_cross_entropy_with_logits(
        d_logits, d_target, pos_weight=det_pos
    )
    loss_p = F.binary_cross_entropy_with_logits(
        p_logits, p_target, pos_weight=p_pos
    )
    return EQT_DETECTION_LOSS_WEIGHT * loss_det + EQT_P_LOSS_WEIGHT * loss_p

@torch.no_grad()
def validation_loss(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for x, p_target, d_target in loader:
        x = x.to(device, non_blocking=True)
        p_target = p_target.to(device, non_blocking=True)
        d_target = d_target.to(device, non_blocking=True)

        if model_name == "phasenet":
            loss = phasenet_loss(model, x, p_target)
        else:
            loss = eqtransformer_loss(model, x, p_target, d_target)

        batch_n = x.size(0)
        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

    return total_loss / max(total_n, 1)

def train_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
) -> pd.DataFrame:

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    best_val = np.inf
    patience = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, p_target, d_target in train_loader:
            x = x.to(device, non_blocking=True)
            p_target = p_target.to(device, non_blocking=True)
            d_target = d_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if model_name == "phasenet":
                loss = phasenet_loss(model, x, p_target)
            else:
                loss = eqtransformer_loss(model, x, p_target, d_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_n = x.size(0)
            train_loss_sum += float(loss.item()) * batch_n
            train_n += batch_n

        train_loss_value = train_loss_sum / max(train_n, 1)
        val_loss_value = validation_loss(model_name, model, val_loader, device)
        elapsed = time.time() - start_time

        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss_value,
                "val_loss": val_loss_value,
                "elapsed_s": elapsed,
            }
        )

        print(
            f"[{model_name}] epoch {epoch:03d}/{epochs} "
            f"train={train_loss_value:.6f} "
            f"val={val_loss_value:.6f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_loss_value < best_val - 1e-6:
            best_val = val_loss_value
            patience = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "seisbench_version": getattr(seisbench, "__version__", "unknown"),
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(
                    f"[{model_name}] early stop at epoch {epoch}; "
                    f"best val={best_val:.6f}",
                    flush=True,
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return pd.DataFrame(history)

@torch.no_grad()
def predict_test(
    model_name: str,
    model: torch.nn.Module,
    test_split: dict,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:

    model.eval()
    model.to(device)

    x_tensor = torch.from_numpy(test_split["x"])
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    picks = []
    peak_probs = []

    for (x,) in loader:
        x = x.to(device, non_blocking=True)

        if model_name == "phasenet":
            prob = model(x, logits=False)
            p_prob = prob[:, 0, :]                       # phases="PN"
        else:
            outputs = model(x, logits=False)
            p_prob = outputs[1]                         # detection, P

        pick = torch.argmax(p_prob, dim=-1)
        peak_prob = torch.gather(p_prob, 1, pick[:, None]).squeeze(1)

        picks.extend(pick.detach().cpu().numpy().astype(int).tolist())
        peak_probs.extend(peak_prob.detach().cpu().numpy().astype(float).tolist())

    result = test_split["meta"].copy()
    result["model"] = model_name
    result["pick_sample"] = np.asarray(picks, dtype=int)
    result["peak_probability"] = np.asarray(peak_probs, dtype=float)

    ref = result["reference_arrival_sample"].to_numpy(dtype=float)
    pred = result["pick_sample"].to_numpy(dtype=float)
    signed_samples = pred - ref
    error_samples = np.abs(signed_samples)

    result["signed_error_samples"] = signed_samples
    result["error_samples"] = error_samples
    result["signed_error_ms"] = signed_samples * DT * 1000.0
    result["error_ms"] = error_samples * DT * 1000.0
    return result

def summarize_probability(raw: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:

    rows = []
    grouped = raw.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), raw)]

    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)

        error = group["error_ms"].to_numpy(dtype=float)
        signed = group["signed_error_ms"].to_numpy(dtype=float)
        finite = np.isfinite(error)

        row = dict(zip(group_columns, keys)) if group_columns else {}
        row["n"] = int(len(group))
        row["n_success"] = int(finite.sum())
        row["success_rate_pct"] = float(100.0 * finite.mean())

        if finite.any():
            ef = error[finite]
            sf = signed[finite]
            row["mae_ms"] = float(np.mean(ef))
            row["median_error_ms"] = float(np.median(ef))
            row["rmse_ms"] = float(np.sqrt(np.mean(ef ** 2)))
            row["mean_signed_error_ms"] = float(np.mean(sf))
        else:
            row["mae_ms"] = np.nan
            row["median_error_ms"] = np.nan
            row["rmse_ms"] = np.nan
            row["mean_signed_error_ms"] = np.nan

        for limit_ms in PROBABILITY_LIMITS_MS:
            label = str(int(limit_ms))
            within = finite & (error <= limit_ms)
            row[f"p_error_le_{label}ms_pct"] = float(100.0 * within.mean())

        rows.append(row)

    return pd.DataFrame(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["phasenet", "eqtransformer"],
        default=["phasenet", "eqtransformer"],
    )
    parser.add_argument("--frequencies", nargs="+", type=int, default=FREQUENCIES)
    parser.add_argument("--snrs", nargs="+", type=float, default=SNR_VALUES)
    parser.add_argument("--train-repeats", type=int, default=TRAIN_REPEATS)
    parser.add_argument("--val-repeats", type=int, default=VAL_REPEATS)
    parser.add_argument("--test-repeats", type=int, default=TEST_REPEATS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--noise-csv",
        type=Path,
        default=None,
        help="Optional real-noise CSV; WGN is used when omitted.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and load the best checkpoint already in output-dir.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.base_seed)
    device = resolve_device(args.device)

    if min(args.train_repeats, args.val_repeats, args.test_repeats) <= 0:
        raise ValueError("train/val/test repeats must all be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    noise_bank = load_real_noise_csv(args.noise_csv) if args.noise_csv else None

    print("=" * 78)
    print("SeisBench PhaseNet / EQTransformer synthetic benchmark")
    print(f"SeisBench version : {getattr(seisbench, '__version__', 'unknown')}")
    print(f"Device            : {device}")
    print(f"Input samples     : {EXPECTED_SAMPLES}")
    print(f"Sampling rate     : {1.0 / DT:.1f} Hz")
    print(f"Frequencies       : {args.frequencies} Hz")
    print(f"SNR               : {args.snrs} dB")
    print(f"Train repeats     : {args.train_repeats} / condition")
    print(f"Val repeats       : {args.val_repeats} / condition")
    print(f"Test repeats      : {args.test_repeats} / condition")
    print(f"Noise             : {'real' if noise_bank is not None else 'WGN'}")
    print(f"Models            : {args.models}")
    print("=" * 78, flush=True)

    print("\nGenerating train split...", flush=True)
    train_split = generate_split(
        "train", args.frequencies, args.snrs, args.train_repeats,
        args.base_seed, noise_bank
    )
    print("Generating validation split...", flush=True)
    val_split = generate_split(
        "val", args.frequencies, args.snrs, args.val_repeats,
        args.base_seed, noise_bank
    )
    print("Generating test split...", flush=True)
    test_split = generate_split(
        "test", args.frequencies, args.snrs, args.test_repeats,
        args.base_seed, noise_bank
    )

    config = {
        "seisbench_version": getattr(seisbench, "__version__", "unknown"),
        "sampling_rate_hz": 1.0 / DT,
        "input_samples": EXPECTED_SAMPLES,
        "input_channels": 1,
        "frequencies_hz": args.frequencies,
        "snr_db": args.snrs,
        "train_repeats_per_condition": args.train_repeats,
        "val_repeats_per_condition": args.val_repeats,
        "test_repeats_per_condition": args.test_repeats,
        "p_label_sigma_ms": P_LABEL_SIGMA_MS,
        "detection_length_ms": DETECTION_LENGTH_MS,
        "reference": "AIC_prepeak_clean",
        "normalization": "per-trace demean + std",
        "noise": "real" if noise_bank is not None else "WGN",
        "base_seed": args.base_seed,
    }
    with open(args.output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    all_raw = []
    all_history = []

    for model_name in args.models:
        set_seed(args.base_seed)

        if model_name == "phasenet":
            model = build_phasenet()
            batch_size = PHASENET_BATCH_SIZE
            lr = PHASENET_LR
            epochs = PHASENET_EPOCHS
        else:
            model = build_eqtransformer()
            batch_size = EQT_BATCH_SIZE
            lr = EQT_LR
            epochs = EQT_EPOCHS

        checkpoint_path = args.output_dir / f"{model_name}_best.pt"

        if args.skip_train:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"\n[{model_name}] loaded: {checkpoint_path}", flush=True)
        else:
            print(f"\nTraining {model_name}...", flush=True)
            train_loader = make_loader(train_split, batch_size, shuffle=True)
            val_loader = make_loader(val_split, batch_size, shuffle=False)
            history = train_model(
                model_name, model, train_loader, val_loader, device,
                epochs, lr, checkpoint_path
            )
            all_history.append(history)

        print(f"\nTesting {model_name}...", flush=True)
        raw = predict_test(model_name, model, test_split, batch_size, device)
        all_raw.append(raw)

        raw.to_csv(
            args.output_dir / f"raw_{model_name}_test.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_all = pd.concat(all_raw, ignore_index=True)

    summary_condition = summarize_probability(
        raw_all, ["model", "frequency_hz", "snr_db"]
    )
    summary_snr = summarize_probability(
        raw_all, ["model", "snr_db"]
    )
    summary_overall = summarize_probability(
        raw_all, ["model"]
    )

    raw_all.to_csv(
        args.output_dir / "raw_phasenet_eqtransformer_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_condition.to_csv(
        args.output_dir / "summary_by_frequency_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_snr.to_csv(
        args.output_dir / "summary_by_snr.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_overall.to_csv(
        args.output_dir / "summary_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if all_history:
        pd.concat(all_history, ignore_index=True).to_csv(
            args.output_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 78)
    print("Summary by frequency and SNR")
    print(summary_condition.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nSummary by SNR (averaged over frequencies)")
    print(summary_snr.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nOverall")
    print(summary_overall.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")
    print("=" * 78, flush=True)

if __name__ == "__main__":
    main()
