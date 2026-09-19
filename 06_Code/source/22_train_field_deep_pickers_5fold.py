from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

import seisbench
import seisbench.models as sbm

DATA_ROOT = Path("./pre_data")
SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"

OUTPUT_DIR = DATA_ROOT / "results_phasenet_eqtransformer_5fold"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

FS = 1000.0
DT = 1.0 / FS
EXPECTED_SAMPLES = 1000

BASE_SEED = 20260917

N_FOLDS = 5

VAL_RATIO_WITHIN_DEV = 0.125

LABEL_IS_ONE_BASED = False

P_LABEL_SIGMA_MS = 3.0

DETECTION_LENGTH_MS = 100.0

PHASENET_BATCH_SIZE = 64
PHASENET_LR = 1e-3
PHASENET_EPOCHS = 100
PHASENET_P_WEIGHT = 8.0

EQT_BATCH_SIZE = 32
EQT_LR = 5e-4
EQT_EPOCHS = 100

EQT_DETECTION_LOSS_WEIGHT = 0.2
EQT_P_LOSS_WEIGHT = 1.0
EQT_DETECTION_POS_WEIGHT = 3.0
EQT_P_POS_WEIGHT = 20.0

EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

PROBABILITY_LIMITS_MS = (2.0, 5.0, 10.0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def load_checkpoint(path, device):

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def read_signal(path):
    data = np.loadtxt(path)

    if data.ndim > 1:
        data = data[:, 0]

    data = np.asarray(data, dtype=np.float32).ravel()

    if len(data) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{path.name}: length={len(data)}, expected={EXPECTED_SAMPLES}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path.name}: contains NaN or Inf")

    return data

def read_label(path):
    data = np.loadtxt(path)

    arrival = int(
        round(
            float(
                np.asarray(data)
                .reshape(-1)[0]
            )
        )
    )

    if LABEL_IS_ONE_BASED:
        arrival -= 1

    if not 0 <= arrival < EXPECTED_SAMPLES:
        raise ValueError(
            f"{path.name}: arrival={arrival}, "
            f" is outside 0~{EXPECTED_SAMPLES - 1}"
        )

    return arrival

def normalize_trace(x):

    x = np.asarray(x, dtype=np.float32).ravel()
    x = x - np.mean(x)

    std = float(np.std(x))

    if std > 1e-8:
        x = x / std

    return x.astype(np.float32, copy=False)

def gaussian_target(n, center, sigma_samples):
    idx = np.arange(n, dtype=np.float32)

    y = np.exp(
        -0.5 *
        ((idx - float(center)) / float(sigma_samples)) ** 2
    )

    return y.astype(np.float32)

def detection_target(n, arrival, length_samples):
    y = np.zeros(n, dtype=np.float32)

    start = max(0, int(arrival))
    end = min(n, start + int(length_samples))

    y[start:end] = 1.0

    return y

def load_real_dataset():
    signal_files = sorted(
        SIGNAL_DIR.glob("*.txt"),
        key=lambda x: int(x.stem)
    )

    print("Signal files:", len(signal_files))

    records = []

    for signal_path in signal_files:
        label_path = LABEL_DIR / signal_path.name

        if not label_path.exists():
            print("Missing label:", signal_path.name)
            continue

        try:
            signal = read_signal(signal_path)
            arrival = read_label(label_path)

            records.append({
                "file_id": int(signal_path.stem),
                "signal": signal,
                "arrival": arrival
            })

        except Exception as e:
            print(signal_path.name, e)

    print("Valid records:", len(records))

    if len(records) < N_FOLDS:
        raise RuntimeError(
            f"Valid record count {len(records)} is insufficient for {N_FOLDS}-fold."
        )

    return records

def build_full_dataset(records):
    n = len(records)

    x_all = np.empty(
        (n, 1, EXPECTED_SAMPLES),
        dtype=np.float32
    )

    p_all = np.empty(
        (n, EXPECTED_SAMPLES),
        dtype=np.float32
    )

    d_all = np.empty(
        (n, EXPECTED_SAMPLES),
        dtype=np.float32
    )

    sigma_samples = P_LABEL_SIGMA_MS / (DT * 1000.0)

    det_samples = max(
        1,
        int(
            round(
                DETECTION_LENGTH_MS /
                (DT * 1000.0)
            )
        )
    )

    meta_rows = []

    for i, record in enumerate(records):
        signal = normalize_trace(record["signal"])
        arrival = int(record["arrival"])

        x_all[i, 0] = signal

        p_all[i] = gaussian_target(
            EXPECTED_SAMPLES,
            arrival,
            sigma_samples
        )

        d_all[i] = detection_target(
            EXPECTED_SAMPLES,
            arrival,
            det_samples
        )

        meta_rows.append({
            "record_index": i,
            "file_id": int(record["file_id"]),
            "reference_arrival_sample": arrival,
            "reference_arrival_ms": arrival * DT * 1000.0
        })

    return {
        "x": x_all,
        "p": p_all,
        "detection": d_all,
        "meta": pd.DataFrame(meta_rows)
    }

def subset_split(full_data, indices, split_name):
    indices = np.asarray(indices, dtype=int)

    meta = (
        full_data["meta"]
        .iloc[indices]
        .copy()
        .reset_index(drop=True)
    )

    meta["split"] = split_name

    return {
        "x": full_data["x"][indices],
        "p": full_data["p"][indices],
        "detection": full_data["detection"][indices],
        "meta": meta
    }

def build_fivefold_splits(n_records):

    indices = np.arange(n_records)

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=BASE_SEED
    )

    folds = []
    assignment_rows = []

    for fold_id, (dev_idx, test_idx) in enumerate(
        kfold.split(indices),
        start=1
    ):
        train_idx, val_idx = train_test_split(
            dev_idx,
            test_size=VAL_RATIO_WITHIN_DEV,
            random_state=BASE_SEED + fold_id,
            shuffle=True
        )

        folds.append({
            "fold": fold_id,
            "train_idx": np.asarray(train_idx, dtype=int),
            "val_idx": np.asarray(val_idx, dtype=int),
            "test_idx": np.asarray(test_idx, dtype=int)
        })

        for idx in train_idx:
            assignment_rows.append({
                "fold": fold_id,
                "record_index": int(idx),
                "role": "train"
            })

        for idx in val_idx:
            assignment_rows.append({
                "fold": fold_id,
                "record_index": int(idx),
                "role": "val"
            })

        for idx in test_idx:
            assignment_rows.append({
                "fold": fold_id,
                "record_index": int(idx),
                "role": "test"
            })

    return folds, pd.DataFrame(assignment_rows)

def make_loader(split, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(split["x"]),
        torch.from_numpy(split["p"]),
        torch.from_numpy(split["detection"])
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

def build_phasenet():
    return sbm.VariableLengthPhaseNet(
        in_samples=EXPECTED_SAMPLES,
        in_channels=1,
        classes=2,
        phases="PN",
        sampling_rate=FS,
        norm="std",
        norm_axis=(-1,),
        output_activation="softmax"
    )

def build_eqtransformer():
    return sbm.EQTransformer(
        in_channels=1,
        in_samples=EXPECTED_SAMPLES,
        classes=1,
        phases="P",
        lstm_blocks=3,
        drop_rate=0.1,
        original_compatible=False,
        sampling_rate=FS,
        norm="std"
    )

def phasenet_loss(model, x, p_target):
    logits = model(x, logits=True)

    target = torch.stack(
        (p_target, 1.0 - p_target),
        dim=1
    )

    log_prob = F.log_softmax(
        logits,
        dim=1
    )

    ce = -(target * log_prob).sum(dim=1)

    time_weight = (
        1.0 +
        (PHASENET_P_WEIGHT - 1.0) * p_target
    )

    return (ce * time_weight).mean()

def eqtransformer_loss(model, x, p_target, d_target):
    outputs = model(x, logits=True)

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError(
            "Unexpected EQTransformer output format; detection and P branches are required."
        )

    d_logits = outputs[0]
    p_logits = outputs[1]

    det_pos = torch.tensor(
        EQT_DETECTION_POS_WEIGHT,
        device=x.device
    )

    p_pos = torch.tensor(
        EQT_P_POS_WEIGHT,
        device=x.device
    )

    loss_det = F.binary_cross_entropy_with_logits(
        d_logits,
        d_target,
        pos_weight=det_pos
    )

    loss_p = F.binary_cross_entropy_with_logits(
        p_logits,
        p_target,
        pos_weight=p_pos
    )

    return (
        EQT_DETECTION_LOSS_WEIGHT * loss_det +
        EQT_P_LOSS_WEIGHT * loss_p
    )

@torch.no_grad()
def validation_loss(model_name, model, loader, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    for x, p_target, d_target in loader:
        x = x.to(device, non_blocking=True)
        p_target = p_target.to(device, non_blocking=True)
        d_target = d_target.to(device, non_blocking=True)

        if model_name == "phasenet":
            loss = phasenet_loss(
                model,
                x,
                p_target
            )
        else:
            loss = eqtransformer_loss(
                model,
                x,
                p_target,
                d_target
            )

        batch_n = x.size(0)

        total_loss += (
            float(loss.item()) *
            batch_n
        )

        total_n += batch_n

    return total_loss / max(total_n, 1)

def train_model(
    model_name,
    fold_id,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    checkpoint_path
):
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
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
                loss = phasenet_loss(
                    model,
                    x,
                    p_target
                )
            else:
                loss = eqtransformer_loss(
                    model,
                    x,
                    p_target,
                    d_target
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            batch_n = x.size(0)

            train_loss_sum += (
                float(loss.item()) *
                batch_n
            )

            train_n += batch_n

        train_loss_value = (
            train_loss_sum /
            max(train_n, 1)
        )

        val_loss_value = validation_loss(
            model_name,
            model,
            val_loader,
            device
        )

        elapsed = time.time() - start_time

        history.append({
            "model": model_name,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss_value,
            "val_loss": val_loss_value,
            "elapsed_s": elapsed
        })

        print(
            f"[{model_name}] "
            f"fold {fold_id}/{N_FOLDS} "
            f"epoch {epoch:03d}/{epochs} "
            f"train={train_loss_value:.6f} "
            f"val={val_loss_value:.6f} "
            f"time={elapsed:.1f}s",
            flush=True
        )

        if val_loss_value < best_val - 1e-6:
            best_val = val_loss_value
            patience = 0

            torch.save({
                "model_name": model_name,
                "fold": fold_id,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val,
                "epoch": epoch,
                "seisbench_version":
                    getattr(
                        seisbench,
                        "__version__",
                        "unknown"
                    )
            }, checkpoint_path)

        else:
            patience += 1

            if patience >= EARLY_STOP_PATIENCE:
                print(
                    f"[{model_name}] "
                    f"fold {fold_id} early stop "
                    f"at epoch {epoch}; "
                    f"best val={best_val:.6f}",
                    flush=True
                )
                break

    checkpoint = load_checkpoint(
        checkpoint_path,
        device
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    return pd.DataFrame(history)

@torch.no_grad()
def predict_test(
    model_name,
    fold_id,
    model,
    test_split,
    batch_size,
    device
):
    model.eval()
    model.to(device)

    dataset = TensorDataset(
        torch.from_numpy(
            test_split["x"]
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    picks = []
    peak_probs = []

    for (x,) in loader:
        x = x.to(
            device,
            non_blocking=True
        )

        if model_name == "phasenet":
            prob = model(
                x,
                logits=False
            )

            p_prob = prob[:, 0, :]

        else:
            outputs = model(
                x,
                logits=False
            )

            p_prob = outputs[1]

        pick = torch.argmax(
            p_prob,
            dim=-1
        )

        peak_prob = torch.gather(
            p_prob,
            1,
            pick[:, None]
        ).squeeze(1)

        picks.extend(
            pick.detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        peak_probs.extend(
            peak_prob.detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

    result = test_split["meta"].copy()

    result["model"] = model_name
    result["fold"] = fold_id

    result["pick_sample"] = np.asarray(
        picks,
        dtype=int
    )

    result["peak_probability"] = np.asarray(
        peak_probs,
        dtype=float
    )

    ref = result[
        "reference_arrival_sample"
    ].to_numpy(dtype=float)

    pred = result[
        "pick_sample"
    ].to_numpy(dtype=float)

    signed_samples = pred - ref
    error_samples = np.abs(signed_samples)

    result["signed_error_samples"] = signed_samples
    result["error_samples"] = error_samples

    result["signed_error_ms"] = (
        signed_samples *
        DT *
        1000.0
    )

    result["error_ms"] = (
        error_samples *
        DT *
        1000.0
    )

    return result

def summarize_probability(raw, group_columns):
    rows = []

    if group_columns:
        grouped = raw.groupby(
            group_columns,
            dropna=False,
            sort=True
        )
    else:
        grouped = [((), raw)]

    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)

        row = (
            dict(zip(group_columns, keys))
            if group_columns
            else {}
        )

        error = group[
            "error_ms"
        ].to_numpy(dtype=float)

        signed = group[
            "signed_error_ms"
        ].to_numpy(dtype=float)

        finite = np.isfinite(error)

        row["n"] = int(len(group))

        if finite.any():
            ef = error[finite]
            sf = signed[finite]

            row["mae_ms"] = float(
                np.mean(ef)
            )

            row["median_error_ms"] = float(
                np.median(ef)
            )

            row["rmse_ms"] = float(
                np.sqrt(
                    np.mean(
                        ef ** 2
                    )
                )
            )

            row["mean_signed_error_ms"] = float(
                np.mean(sf)
            )

        else:
            row["mae_ms"] = np.nan
            row["median_error_ms"] = np.nan
            row["rmse_ms"] = np.nan
            row["mean_signed_error_ms"] = np.nan

        for limit_ms in PROBABILITY_LIMITS_MS:
            name = str(int(limit_ms))

            row[
                f"p_error_le_{name}ms_pct"
            ] = float(
                100.0 *
                np.mean(
                    finite &
                    (error <= limit_ms)
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_fold_mean_std(summary_fold):

    metrics = [
        "mae_ms",
        "median_error_ms",
        "rmse_ms",
        "mean_signed_error_ms",
        "p_error_le_2ms_pct",
        "p_error_le_5ms_pct",
        "p_error_le_10ms_pct"
    ]

    rows = []

    for model_name, group in summary_fold.groupby("model"):
        row = {
            "model": model_name,
            "folds": len(group)
        }

        for metric in metrics:
            values = group[
                metric
            ].to_numpy(dtype=float)

            row[f"{metric}_mean"] = float(
                np.mean(values)
            )

            row[f"{metric}_std"] = float(
                np.std(
                    values,
                    ddof=1
                )
            ) if len(values) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)

def validate_oof_results(raw_all, total_records, models):

    for model_name in models:
        part = raw_all[
            raw_all["model"] == model_name
        ]

        if len(part) != total_records:
            raise RuntimeError(
                f"{model_name}: OOF result count="
                f"{len(part)}，expected={total_records}"
            )

        if part["file_id"].nunique() != total_records:
            duplicated = part[
                part["file_id"].duplicated(
                    keep=False
                )
            ]["file_id"].tolist()

            raise RuntimeError(
                f"{model_name}: OOF contains duplicate or missing file_id values: "
                f"{duplicated[:20]}"
            )

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "phasenet",
            "eqtransformer"
        ],
        default=[
            "phasenet",
            "eqtransformer"
        ]
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and load existing five-fold checkpoints."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    set_seed(BASE_SEED)

    device = resolve_device(
        args.device
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("PhaseNet / EQTransformer 5-fold OOF benchmark")
    print("=" * 80)

    print(
        "SeisBench version:",
        getattr(
            seisbench,
            "__version__",
            "unknown"
        )
    )

    print("Device:", device)
    print("Sampling rate:", FS, "Hz")
    print("Input samples:", EXPECTED_SAMPLES)
    print("Outer folds:", N_FOLDS)
    print("Models:", args.models)

    print("=" * 80)

    records = load_real_dataset()
    total_records = len(records)

    full_data = build_full_dataset(
        records
    )

    folds, assignment = build_fivefold_splits(
        total_records
    )

    id_map = full_data[
        "meta"
    ].set_index(
        "record_index"
    )["file_id"]

    assignment["file_id"] = (
        assignment["record_index"]
        .map(id_map)
        .astype(int)
    )

    assignment = assignment[
        [
            "fold",
            "record_index",
            "file_id",
            "role"
        ]
    ]

    assignment.to_csv(
        OUTPUT_DIR /
        "dataset_5fold_split.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()

    for fold in folds:
        print(
            f"Fold {fold['fold']}: "
            f"train={len(fold['train_idx'])}, "
            f"val={len(fold['val_idx'])}, "
            f"test={len(fold['test_idx'])}"
        )

    config = {
        "data_root":
            str(DATA_ROOT.resolve()),

        "seisbench_version":
            getattr(
                seisbench,
                "__version__",
                "unknown"
            ),

        "sampling_rate_hz":
            FS,

        "input_samples":
            EXPECTED_SAMPLES,

        "input_channels":
            1,

        "total_records":
            total_records,

        "outer_folds":
            N_FOLDS,

        "outer_test_fraction":
            1.0 / N_FOLDS,

        "validation_fraction_within_dev":
            VAL_RATIO_WITHIN_DEV,

        "approx_total_train_fraction":
            (
                1.0 - 1.0 / N_FOLDS
            ) * (
                1.0 - VAL_RATIO_WITHIN_DEV
            ),

        "approx_total_val_fraction":
            (
                1.0 - 1.0 / N_FOLDS
            ) * VAL_RATIO_WITHIN_DEV,

        "p_label_sigma_ms":
            P_LABEL_SIGMA_MS,

        "detection_length_ms":
            DETECTION_LENGTH_MS,

        "normalization":
            "per-trace demean + std",

        "label_indexing":
            "0-based" if not LABEL_IS_ONE_BASED
            else "1-based converted to 0-based",

        "base_seed":
            BASE_SEED
    }

    with open(
        OUTPUT_DIR /
        "experiment_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )

    all_raw = []
    all_history = []

    for model_name in args.models:

        print()
        print("#" * 80)
        print(f"MODEL: {model_name}")
        print("#" * 80)

        model_fold_raw = []

        for fold in folds:
            fold_id = fold["fold"]

            print()
            print("=" * 80)
            print(
                f"{model_name.upper()} "
                f"FOLD {fold_id}/{N_FOLDS}"
            )
            print("=" * 80)

            model_seed = (
                BASE_SEED +
                fold_id * 1000 +
                (
                    0 if model_name == "phasenet"
                    else 100
                )
            )

            set_seed(model_seed)

            train_split = subset_split(
                full_data,
                fold["train_idx"],
                "train"
            )

            val_split = subset_split(
                full_data,
                fold["val_idx"],
                "val"
            )

            test_split = subset_split(
                full_data,
                fold["test_idx"],
                "test"
            )

            print(
                f"train={len(train_split['meta'])}, "
                f"val={len(val_split['meta'])}, "
                f"test={len(test_split['meta'])}"
            )

            if model_name == "phasenet":
                model = build_phasenet()
                batch_size = PHASENET_BATCH_SIZE
                learning_rate = PHASENET_LR
                epochs = PHASENET_EPOCHS

            else:
                model = build_eqtransformer()
                batch_size = EQT_BATCH_SIZE
                learning_rate = EQT_LR
                epochs = EQT_EPOCHS

            checkpoint_path = (
                CHECKPOINT_DIR /
                f"{model_name}_fold{fold_id}_best.pt"
            )

            if args.skip_train:
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found: "
                        f"{checkpoint_path}"
                    )

                checkpoint = load_checkpoint(
                    checkpoint_path,
                    device
                )

                model.load_state_dict(
                    checkpoint["state_dict"]
                )

                print(
                    f"[{model_name}] "
                    f"fold {fold_id} loaded: "
                    f"{checkpoint_path}",
                    flush=True
                )

            else:
                train_loader = make_loader(
                    train_split,
                    batch_size,
                    shuffle=True
                )

                val_loader = make_loader(
                    val_split,
                    batch_size,
                    shuffle=False
                )

                history = train_model(
                    model_name=model_name,
                    fold_id=fold_id,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    checkpoint_path=checkpoint_path
                )

                all_history.append(
                    history
                )

            print(
                f"\nTesting "
                f"{model_name} "
                f"fold {fold_id}...",
                flush=True
            )

            raw = predict_test(
                model_name=model_name,
                fold_id=fold_id,
                model=model,
                test_split=test_split,
                batch_size=batch_size,
                device=device
            )

            model_fold_raw.append(
                raw
            )

            all_raw.append(
                raw
            )

            raw.to_csv(
                OUTPUT_DIR /
                f"raw_{model_name}_fold{fold_id}_test.csv",
                index=False,
                encoding="utf-8-sig"
            )

            fold_summary = summarize_probability(
                raw,
                ["model", "fold"]
            )

            print()
            print(
                fold_summary.to_string(
                    index=False,
                    float_format=
                    lambda x: f"{x:.3f}"
                )
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_oof = pd.concat(
            model_fold_raw,
            ignore_index=True
        )

        model_oof = (
            model_oof
            .sort_values("file_id")
            .reset_index(drop=True)
        )

        model_oof.to_csv(
            OUTPUT_DIR /
            f"raw_{model_name}_oof_1000.csv",
            index=False,
            encoding="utf-8-sig"
        )

    raw_all = pd.concat(
        all_raw,
        ignore_index=True
    )

    raw_all = (
        raw_all
        .sort_values(
            ["model", "file_id"]
        )
        .reset_index(drop=True)
    )

    validate_oof_results(
        raw_all,
        total_records,
        args.models
    )

    summary_by_fold = summarize_probability(
        raw_all,
        ["model", "fold"]
    )

    summary_overall = summarize_probability(
        raw_all,
        ["model"]
    )

    summary_fold_mean_std = summarize_fold_mean_std(
        summary_by_fold
    )

    raw_all.to_csv(
        OUTPUT_DIR /
        "raw_all_models_oof.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_by_fold.to_csv(
        OUTPUT_DIR /
        "summary_by_fold.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_overall.to_csv(
        OUTPUT_DIR /
        "summary_oof_overall.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_fold_mean_std.to_csv(
        OUTPUT_DIR /
        "summary_fold_mean_std.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if all_history:
        pd.concat(
            all_history,
            ignore_index=True
        ).to_csv(
            OUTPUT_DIR /
            "training_history.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print()
    print("=" * 80)
    print("5-FOLD RESULTS")
    print("=" * 80)

    print("\nResults by fold:")
    print(
        summary_by_fold.to_string(
            index=False,
            float_format=
            lambda x: f"{x:.3f}"
        )
    )

    print()
    print("=" * 80)
    print("OOF OVERALL RESULTS")
    print(
        f"Each model contains "
        f"{total_records} independent test predictions."
    )
    print("=" * 80)

    print(
        summary_overall.to_string(
            index=False,
            float_format=
            lambda x: f"{x:.3f}"
        )
    )

    print()
    print("=" * 80)
    print("5-FOLD MEAN ± STD")
    print("=" * 80)

    print(
        summary_fold_mean_std.to_string(
            index=False,
            float_format=
            lambda x: f"{x:.3f}"
        )
    )

    print()
    print("Results saved to:")
    print(OUTPUT_DIR.resolve())
    print("=" * 80)

if __name__ == "__main__":
    main()
