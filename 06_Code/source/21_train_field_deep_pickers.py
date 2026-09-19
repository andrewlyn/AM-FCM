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
from torch.utils.data import DataLoader, TensorDataset

import seisbench
import seisbench.models as sbm

DATA_ROOT = Path("./pre_data")
SIGNAL_DIR = DATA_ROOT / "signal"
LABEL_DIR = DATA_ROOT / "labels"

OUTPUT_DIR = DATA_ROOT / "results_phasenet_eqtransformer"

FS = 1000.0
DT = 1.0 / FS
EXPECTED_SAMPLES = 1000

BASE_SEED = 20260916

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
        return torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(name)

def read_signal(path):
    data = np.loadtxt(path)

    if data.ndim > 1:
        data = data[:, 0]

    data = np.asarray(
        data,
        dtype=np.float32
    ).ravel()

    if len(data) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{path.name}: "
            f"length={len(data)}, "
            f"expected={EXPECTED_SAMPLES}"
        )

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
            f"{path.name}: "
            f"arrival label={arrival}, "
            f" is outside 0~{EXPECTED_SAMPLES - 1}"
        )

    return arrival

def normalize_trace(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x - np.mean(x)

    std = float(np.std(x))

    if std > 1e-8:
        x = x / std

    return x.astype(
        np.float32,
        copy=False
    )

def gaussian_target(n, center, sigma_samples):
    idx = np.arange(
        n,
        dtype=np.float32
    )

    y = np.exp(
        -0.5 *
        (
            (idx - float(center))
            /
            float(sigma_samples)
        ) ** 2
    )

    return y.astype(
        np.float32
    )

def detection_target(
    n,
    arrival,
    length_samples
):

    y = np.zeros(
        n,
        dtype=np.float32
    )

    start = max(
        0,
        int(arrival)
    )

    end = min(
        n,
        start + int(length_samples)
    )

    y[start:end] = 1.0

    return y

def load_real_dataset():

    signal_files = sorted(
        SIGNAL_DIR.glob("*.txt"),
        key=lambda x: int(x.stem)
    )

    print(
        "Signal files:",
        len(signal_files)
    )

    records = []

    for signal_path in signal_files:

        label_path = (
            LABEL_DIR /
            signal_path.name
        )

        if not label_path.exists():

            print(
                "Missing label:",
                signal_path.name
            )

            continue

        try:

            signal = read_signal(
                signal_path
            )

            arrival = read_label(
                label_path
            )

            records.append(
                {
                    "file_id": signal_path.stem,
                    "signal": signal,
                    "arrival": arrival
                }
            )

        except Exception as e:

            print(
                signal_path.name,
                e
            )

    print(
        "Valid records:",
        len(records)
    )

    if len(records) == 0:
        raise RuntimeError(
            "No valid records found."
        )

    return records

def split_records(records, seed):

    n = len(records)

    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        n
    )

    n_train = int(
        n * TRAIN_RATIO
    )

    n_val = int(
        n * VAL_RATIO
    )

    train_idx = indices[
        :n_train
    ]

    val_idx = indices[
        n_train:
        n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    train_records = [
        records[i]
        for i in train_idx
    ]

    val_records = [
        records[i]
        for i in val_idx
    ]

    test_records = [
        records[i]
        for i in test_idx
    ]

    print()
    print(
        "Train:",
        len(train_records)
    )

    print(
        "Validation:",
        len(val_records)
    )

    print(
        "Test:",
        len(test_records)
    )

    return (
        train_records,
        val_records,
        test_records
    )

def build_split(
    records,
    split_name
):

    n = len(records)

    x_all = np.empty(
        (
            n,
            1,
            EXPECTED_SAMPLES
        ),
        dtype=np.float32
    )

    p_all = np.empty(
        (
            n,
            EXPECTED_SAMPLES
        ),
        dtype=np.float32
    )

    d_all = np.empty(
        (
            n,
            EXPECTED_SAMPLES
        ),
        dtype=np.float32
    )

    sigma_samples = (
        P_LABEL_SIGMA_MS
        /
        (DT * 1000.0)
    )

    det_samples = max(
        1,
        int(
            round(
                DETECTION_LENGTH_MS
                /
                (DT * 1000.0)
            )
        )
    )

    meta = []

    for i, record in enumerate(
        records
    ):

        signal = normalize_trace(
            record["signal"]
        )

        arrival = int(
            record["arrival"]
        )

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

        meta.append(
            {
                "split": split_name,
                "file_id": record["file_id"],
                "reference_arrival_sample": arrival,
                "reference_arrival_ms":
                    arrival * DT * 1000.0
            }
        )

    return {
        "x": x_all,
        "p": p_all,
        "detection": d_all,
        "meta": pd.DataFrame(meta)
    }

def make_loader(
    split,
    batch_size,
    shuffle
):

    dataset = TensorDataset(
        torch.from_numpy(
            split["x"]
        ),
        torch.from_numpy(
            split["p"]
        ),
        torch.from_numpy(
            split["detection"]
        )
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

def phasenet_loss(
    model,
    x,
    p_target
):

    logits = model(
        x,
        logits=True
    )

    target = torch.stack(
        (
            p_target,
            1.0 - p_target
        ),
        dim=1
    )

    log_prob = F.log_softmax(
        logits,
        dim=1
    )

    ce = -(
        target * log_prob
    ).sum(
        dim=1
    )

    time_weight = (
        1.0
        +
        (
            PHASENET_P_WEIGHT - 1.0
        )
        *
        p_target
    )

    return (
        ce *
        time_weight
    ).mean()

def eqtransformer_loss(
    model,
    x,
    p_target,
    d_target
):

    outputs = model(
        x,
        logits=True
    )

    if not isinstance(
        outputs,
        (tuple, list)
    ) or len(outputs) < 2:

        raise RuntimeError(
            "Unexpected EQTransformer output format."
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

    loss_det = (
        F.binary_cross_entropy_with_logits(
            d_logits,
            d_target,
            pos_weight=det_pos
        )
    )

    loss_p = (
        F.binary_cross_entropy_with_logits(
            p_logits,
            p_target,
            pos_weight=p_pos
        )
    )

    return (
        EQT_DETECTION_LOSS_WEIGHT
        *
        loss_det
        +
        EQT_P_LOSS_WEIGHT
        *
        loss_p
    )

@torch.no_grad()
def validation_loss(
    model_name,
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0
    total_n = 0

    for (
        x,
        p_target,
        d_target
    ) in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        p_target = p_target.to(
            device,
            non_blocking=True
        )

        d_target = d_target.to(
            device,
            non_blocking=True
        )

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
            float(loss.item())
            *
            batch_n
        )

        total_n += batch_n

    return (
        total_loss
        /
        max(total_n, 1)
    )

def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    checkpoint_path
):

    model = model.to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    best_val = np.inf
    patience = 0

    history = []

    start_time = time.time()

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for (
            x,
            p_target,
            d_target
        ) in train_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            p_target = p_target.to(
                device,
                non_blocking=True
            )

            d_target = d_target.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

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
                float(loss.item())
                *
                batch_n
            )

            train_n += batch_n

        train_loss_value = (
            train_loss_sum
            /
            max(train_n, 1)
        )

        val_loss_value = (
            validation_loss(
                model_name,
                model,
                val_loader,
                device
            )
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss":
                    train_loss_value,
                "val_loss":
                    val_loss_value,
                "elapsed_s":
                    elapsed
            }
        )

        print(
            f"[{model_name}] "
            f"epoch {epoch:03d}/{epochs} "
            f"train={train_loss_value:.6f} "
            f"val={val_loss_value:.6f} "
            f"time={elapsed:.1f}s",
            flush=True
        )

        if (
            val_loss_value
            <
            best_val - 1e-6
        ):

            best_val = (
                val_loss_value
            )

            patience = 0

            torch.save(
                {
                    "model_name":
                        model_name,
                    "state_dict":
                        model.state_dict(),
                    "best_val_loss":
                        best_val,
                    "epoch":
                        epoch,
                    "seisbench_version":
                        getattr(
                            seisbench,
                            "__version__",
                            "unknown"
                        )
                },
                checkpoint_path
            )

        else:

            patience += 1

            if (
                patience
                >=
                EARLY_STOP_PATIENCE
            ):

                print(
                    f"[{model_name}] "
                    f"early stop at "
                    f"epoch {epoch}; "
                    f"best val="
                    f"{best_val:.6f}",
                    flush=True
                )

                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    return pd.DataFrame(
        history
    )

@torch.no_grad()
def predict_test(
    model_name,
    model,
    test_split,
    batch_size,
    device
):

    model.eval()

    model.to(
        device
    )

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

            p_prob = prob[
                :, 0, :
            ]

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

        peak_prob = (
            torch.gather(
                p_prob,
                1,
                pick[:, None]
            )
            .squeeze(1)
        )

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

    result = (
        test_split["meta"]
        .copy()
    )

    result["model"] = (
        model_name
    )

    result["pick_sample"] = (
        np.asarray(
            picks,
            dtype=int
        )
    )

    result[
        "peak_probability"
    ] = np.asarray(
        peak_probs,
        dtype=float
    )

    ref = result[
        "reference_arrival_sample"
    ].to_numpy(
        dtype=float
    )

    pred = result[
        "pick_sample"
    ].to_numpy(
        dtype=float
    )

    signed_samples = (
        pred - ref
    )

    error_samples = (
        np.abs(
            signed_samples
        )
    )

    result[
        "signed_error_samples"
    ] = signed_samples

    result[
        "error_samples"
    ] = error_samples

    result[
        "signed_error_ms"
    ] = (
        signed_samples
        *
        DT
        *
        1000.0
    )

    result[
        "error_ms"
    ] = (
        error_samples
        *
        DT
        *
        1000.0
    )

    return result

def summarize(
    raw
):

    error = raw[
        "error_ms"
    ].to_numpy(
        dtype=float
    )

    signed = raw[
        "signed_error_ms"
    ].to_numpy(
        dtype=float
    )

    rows = []

    for model_name, group in (
        raw.groupby(
            "model"
        )
    ):

        error = group[
            "error_ms"
        ].to_numpy(
            dtype=float
        )

        signed = group[
            "signed_error_ms"
        ].to_numpy(
            dtype=float
        )

        row = {
            "model":
                model_name,

            "n":
                len(group),

            "mae_ms":
                float(
                    np.mean(error)
                ),

            "median_error_ms":
                float(
                    np.median(error)
                ),

            "rmse_ms":
                float(
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    )
                ),

            "mean_signed_error_ms":
                float(
                    np.mean(signed)
                )
        }

        for limit in (
            PROBABILITY_LIMITS_MS
        ):

            name = str(
                int(limit)
            )

            row[
                f"p_error_le_{name}ms_pct"
            ] = float(
                100.0
                *
                np.mean(
                    error
                    <=
                    limit
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
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
        action="store_true"
    )

    return parser.parse_args()

def main():

    args = parse_args()

    set_seed(
        BASE_SEED
    )

    device = resolve_device(
        args.device
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 78)
    print(
        "PhaseNet / EQTransformer "
        "real microseismic benchmark"
    )
    print(
        "SeisBench version:",
        getattr(
            seisbench,
            "__version__",
            "unknown"
        )
    )
    print(
        "Device:",
        device
    )
    print(
        "Sampling rate:",
        FS,
        "Hz"
    )
    print(
        "Input samples:",
        EXPECTED_SAMPLES
    )
    print(
        "Models:",
        args.models
    )
    print("=" * 78)

    records = (
        load_real_dataset()
    )

    (
        train_records,
        val_records,
        test_records
    ) = split_records(
        records,
        BASE_SEED
    )

    split_rows = []

    for name, subset in [
        (
            "train",
            train_records
        ),
        (
            "val",
            val_records
        ),
        (
            "test",
            test_records
        )
    ]:

        for r in subset:

            split_rows.append(
                {
                    "file_id":
                        r["file_id"],
                    "split":
                        name,
                    "arrival":
                        r["arrival"]
                }
            )

    pd.DataFrame(
        split_rows
    ).to_csv(
        OUTPUT_DIR /
        "dataset_split.csv",
        index=False,
        encoding="utf-8-sig"
    )

    train_split = build_split(
        train_records,
        "train"
    )

    val_split = build_split(
        val_records,
        "val"
    )

    test_split = build_split(
        test_records,
        "test"
    )

    config = {
        "data_root":
            str(
                DATA_ROOT.resolve()
            ),

        "sampling_rate_hz":
            FS,

        "input_samples":
            EXPECTED_SAMPLES,

        "input_channels":
            1,

        "train_ratio":
            TRAIN_RATIO,

        "val_ratio":
            VAL_RATIO,

        "test_ratio":
            TEST_RATIO,

        "train_n":
            len(train_records),

        "val_n":
            len(val_records),

        "test_n":
            len(test_records),

        "normalization":
            "per-trace demean + std",

        "p_label_sigma_ms":
            P_LABEL_SIGMA_MS,

        "detection_length_ms":
            DETECTION_LENGTH_MS,

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

    for model_name in (
        args.models
    ):

        set_seed(
            BASE_SEED
        )

        if (
            model_name
            ==
            "phasenet"
        ):

            model = (
                build_phasenet()
            )

            batch_size = (
                PHASENET_BATCH_SIZE
            )

            lr = (
                PHASENET_LR
            )

            epochs = (
                PHASENET_EPOCHS
            )

        else:

            model = (
                build_eqtransformer()
            )

            batch_size = (
                EQT_BATCH_SIZE
            )

            lr = (
                EQT_LR
            )

            epochs = (
                EQT_EPOCHS
            )

        checkpoint_path = (
            OUTPUT_DIR /
            f"{model_name}_best.pt"
        )

        if args.skip_train:

            if not (
                checkpoint_path.exists()
            ):

                raise FileNotFoundError(
                    f"Checkpoint not found: "
                    f"{checkpoint_path}"
                )

            checkpoint = (
                torch.load(
                    checkpoint_path,
                    map_location=device
                )
            )

            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

            print(
                f"\n[{model_name}] "
                f"loaded checkpoint."
            )

        else:

            print(
                f"\nTraining "
                f"{model_name}...",
                flush=True
            )

            train_loader = (
                make_loader(
                    train_split,
                    batch_size,
                    True
                )
            )

            val_loader = (
                make_loader(
                    val_split,
                    batch_size,
                    False
                )
            )

            history = (
                train_model(
                    model_name,
                    model,
                    train_loader,
                    val_loader,
                    device,
                    epochs,
                    lr,
                    checkpoint_path
                )
            )

            all_history.append(
                history
            )

        print(
            f"\nTesting "
            f"{model_name}...",
            flush=True
        )

        raw = predict_test(
            model_name,
            model,
            test_split,
            batch_size,
            device
        )

        all_raw.append(
            raw
        )

        raw.to_csv(
            OUTPUT_DIR /
            f"raw_{model_name}_test.csv",
            index=False,
            encoding="utf-8-sig"
        )

        del model

        if (
            torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    raw_all = pd.concat(
        all_raw,
        ignore_index=True
    )

    summary = summarize(
        raw_all
    )

    raw_all.to_csv(
        OUTPUT_DIR /
        "raw_test_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUTPUT_DIR /
        "summary_test.csv",
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
    print("=" * 78)
    print("TEST RESULTS")
    print("=" * 78)

    print(
        summary.to_string(
            index=False,
            float_format=
            lambda x: f"{x:.3f}"
        )
    )

    print()
    print(
        "Results:",
        OUTPUT_DIR.resolve()
    )

    print("=" * 78)

if __name__ == "__main__":
    main()
