#!/usr/bin/env python3
"""Train and evaluate one MMAUD RGB-to-3D-prism regression model."""

import argparse
import csv
import json
import os
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mmaud3d.data import (
    PrismImageDataset,
)
from mmaud3d.metrics import coordinate_statistics, localization_metrics
from mmaud3d.models import DEFAULT_CHECKPOINTS, create_model


CSV_FIELDS = [
    "experiment_id", "timestamp_utc", "status", "model_name", "model_variant",
    "checkpoint_version", "training_machine", "gpu_model", "cpu_model", "ram_gb",
    "epochs", "best_epoch", "learning_rate", "batch_size", "optimizer",
    "weight_decay", "loss_function", "image_size", "random_seed",
    "pretrained_layers_frozen_epochs",
    "dataset_name", "dataset_version", "modality", "train_split", "val_split",
    "test_split", "coordinate_frame", "coordinate_unit",
    "total_parameters", "trainable_parameters", "training_time_seconds",
    "inference_latency_ms_per_frame", "inference_fps",
    "peak_gpu_memory_allocated_gb", "peak_gpu_memory_reserved_gb",
    "mean_euclidean_error_m", "median_error_m", "p95_error_m",
    "mae_x_m", "mae_y_m", "mae_z_m", "rmse_3d_m", "max_error_m",
    "train_samples", "val_samples", "test_samples", "config_path",
    "checkpoint_path", "notes",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MMAUD V1 RGB-to-3D-prism regression training"
    )
    parser.add_argument(
        "--model", required=True, choices=("yolo", "detr", "dinov2", "dino-drone")
    )
    default_data_root = (
        r"I:\Data\Anti-UAV\MMAUD\v1"
        if os.name == "nt"
        else "/mnt/i/Data/Anti-UAV/MMAUD/v1"
    )
    parser.add_argument(
        "--data_path",
        "--data-root",
        dest="data_root",
        default=default_data_root,
        help="Dataset root; --data-root is an equivalent alias.",
    )
    parser.add_argument("--val", required=True,
                        help="Comma-separated validation recording names, e.g. avata.")
    parser.add_argument("--test", required=True,
                        help="Comma-separated test recording names, e.g. m300.")
    parser.add_argument("--max-sync-delta-ms", type=float, default=100.0,
                        help="Maximum image-to-Leica timestamp difference.")
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrained checkpoint; defaults depend on --model.")
    parser.add_argument("--output-dir", default="runs/mmaud3d")
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Results CSV; defaults to <model>_<test>_<month>_<year>.csv.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many consecutive epochs without improved validation loss.",
    )
    parser.add_argument("--freeze-pretrained-epochs", type=int, default=5,
                        help="Freeze pretrained layers for this many initial epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dataset-version", default="V1")
    parser.add_argument("--coordinate-frame", default="leica_relative",
                        help="Declared frame of x/y/z labels; must be consistent.")
    parser.add_argument("--coordinate-unit", default="m")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive.")
    return args


def _sequence_key(name):
    key = "".join(char.lower() for char in name if char.isalnum())
    if key in ("pham4", "phantom4"):
        return "phantom4"
    return key


def _parse_sequence_names(value, option):
    names = [_sequence_key(item.strip()) for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError(f"{option} must contain at least one recording name.")
    if len(set(names)) != len(names):
        raise ValueError(f"{option} contains duplicate recording names.")
    return set(names)


def _default_results_csv(model_name, test_sequences, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc)
    test_slug = "_".join(sorted(_parse_sequence_names(test_sequences, "--test")))
    month_year = timestamp.strftime("%m_%Y")
    return Path("results") / f"{model_name.lower()}_{test_slug}_{month_year}.csv"


def _read_leica_relative_points(sequence_dir):
    relative_dir = sequence_dir / "leica_point_relative"
    timestamps_path = relative_dir / "timestamps.csv"
    decoded_dir = relative_dir / "decoded"
    if not timestamps_path.is_file():
        raise FileNotFoundError(
            f"Missing relative Leica timestamp index: {timestamps_path}"
        )

    with timestamps_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"timestamp_ns", "decoded_file"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{timestamps_path} must contain timestamp_ns and decoded_file columns."
            )
        indexed_files = []
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp_ns = int(row["timestamp_ns"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid timestamp_ns at {timestamps_path}:{line_number}."
                ) from exc
            decoded_file = (row["decoded_file"] or "").strip()
            if not decoded_file:
                raise ValueError(
                    f"Missing decoded_file at {timestamps_path}:{line_number}."
                )
            point_path = relative_dir / decoded_file
            if not point_path.is_file():
                point_path = decoded_dir / Path(decoded_file).name
            if not point_path.is_file():
                raise FileNotFoundError(
                    f"Leica annotation listed at {timestamps_path}:{line_number} "
                    f"does not exist: {point_path}"
                )
            indexed_files.append((timestamp_ns, point_path))

    indexed_files.sort(key=lambda item: item[0])
    if not indexed_files:
        raise ValueError(f"No Leica relative annotations listed in {timestamps_path}.")

    timestamps = np.asarray([item[0] for item in indexed_files], dtype=np.int64)
    points = []
    for _, point_path in indexed_files:
        with point_path.open(encoding="utf-8") as file:
            annotation = json.load(file)["annotation"]["point"]
        if isinstance(annotation, dict):
            point = [annotation[axis] for axis in ("x", "y", "z")]
        elif isinstance(annotation, (list, tuple)) and len(annotation) == 3:
            point = annotation
        else:
            raise ValueError(
                f"Expected a 3D annotation.point in {point_path}, got {annotation!r}."
            )
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid finite XYZ point in {point_path}: {point!r}.")
        points.append(point.astype(np.float32))
    return timestamps, points


def _image_timestamp_ns(image_path):
    try:
        return int(image_path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Image filename must end in _<timestamp_ns>.png: {image_path}"
        ) from exc


def _resolve_data_root(data_root, windows_host=None):
    """Translate a Windows drive path to its usual WSL mount path."""
    if windows_host is None:
        windows_host = os.name == "nt"
    raw_path = os.fspath(data_root)
    is_windows_drive_path = (
        len(raw_path) >= 3 and raw_path[1] == ":" and raw_path[2] in "\\/"
    )
    if not windows_host and is_windows_drive_path:
        drive = raw_path[0].lower()
        remainder = raw_path[3:].replace("\\", "/")
        return Path("/mnt") / drive / remainder
    return Path(raw_path).expanduser()


def build_dataset_rows(
    data_root,
    val_sequences,
    test_sequences,
    max_sync_delta_ms,
    train_split="train",
    val_split="val",
    test_split="test",
):
    """Pair decoded RGB frames with nearest relative-Leica prism annotations."""
    root = _resolve_data_root(data_root).resolve()
    if (root / "v1").is_dir():
        root = root / "v1"
    if not root.is_dir():
        raise NotADirectoryError(f"MMAUD data directory does not exist: {root}")
    if not np.isfinite(max_sync_delta_ms) or max_sync_delta_ms < 0:
        raise ValueError("--max-sync-delta-ms must be a finite non-negative number.")
    if len({train_split, val_split, test_split}) != 3:
        raise ValueError("Train, validation, and test split labels must be distinct.")

    image_dirs = {
        child.name: child / "usb_cam_image_raw" / "decoded"
        for child in root.iterdir()
        if child.is_dir()
        and (child / "usb_cam_image_raw" / "decoded").is_dir()
    }
    if not image_dirs:
        raise FileNotFoundError(
            f"No <recording>/usb_cam_image_raw/decoded folders found under {root}."
        )

    leica_dirs = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        sequence_key = _sequence_key(child.name.rsplit("_", 1)[-1])
        timestamp_index = child / "leica_point_relative" / "timestamps.csv"
        if timestamp_index.is_file():
            if sequence_key in leica_dirs:
                raise ValueError(
                    f"Multiple relative-Leica folders match recording "
                    f"{child.name.rsplit('_', 1)[-1]!r}."
                )
            leica_dirs[sequence_key] = child

    image_sequence_keys = {_sequence_key(name) for name in image_dirs}
    val_keys = _parse_sequence_names(val_sequences, "--val")
    test_keys = _parse_sequence_names(test_sequences, "--test")
    if val_keys & test_keys:
        raise ValueError("--val and --test recordings must not overlap.")
    unknown = (val_keys | test_keys) - image_sequence_keys
    if unknown:
        raise ValueError(
            "Unknown --val/--test recording(s): "
            + ", ".join(sorted(unknown))
            + ". Available: "
            + ", ".join(sorted(image_dirs))
        )
    train_keys = image_sequence_keys - val_keys - test_keys
    if not train_keys:
        raise ValueError("At least one recording must remain for training.")

    missing_ground_truth = image_sequence_keys - set(leica_dirs)
    if missing_ground_truth:
        raise FileNotFoundError(
            "No matching *_<recording>/leica_point_relative/timestamps.csv "
            "folder found for: "
            + ", ".join(sorted(missing_ground_truth))
        )

    delta_limit_ns = int(max_sync_delta_ms * 1_000_000)
    split_by_key = {
        **{key: train_split for key in train_keys},
        **{key: val_split for key in val_keys},
        **{key: test_split for key in test_keys},
    }
    rows = []
    unmatched = {}
    point_cache = {}
    for sequence, image_dir in image_dirs.items():
        sequence_key = _sequence_key(sequence)
        leica_dir = leica_dirs[sequence_key]
        if sequence_key not in point_cache:
            point_cache[sequence_key] = _read_leica_relative_points(leica_dir)
        leica_timestamps, leica_points = point_cache[sequence_key]

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if not images:
            raise FileNotFoundError(f"No decoded PNG images found in {image_dir}.")
        dropped = 0
        for image_path in images:
            image_timestamp = _image_timestamp_ns(image_path)
            index = int(np.searchsorted(leica_timestamps, image_timestamp))
            candidates = [
                candidate for candidate in (index - 1, index)
                if 0 <= candidate < len(leica_timestamps)
            ]
            nearest = min(
                candidates,
                key=lambda candidate: abs(
                    int(leica_timestamps[candidate]) - image_timestamp
                ),
            )
            if abs(int(leica_timestamps[nearest]) - image_timestamp) > delta_limit_ns:
                dropped += 1
                continue
            x, y, z = leica_points[nearest]
            rows.append({
                "image_path": str(image_path),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "sequence": sequence,
                "split": split_by_key[sequence_key],
            })
        unmatched[sequence] = dropped

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in (train_split, val_split, test_split)
    }
    empty_splits = [split for split, count in counts.items() if count == 0]
    if empty_splits:
        raise ValueError(
            "No images matched relative Leica annotations within "
            f"{max_sync_delta_ms:g} ms for split(s): {', '.join(empty_splits)}."
        )
    print(
        f"Loaded {len(rows)} synchronized images from {root}: "
        + ", ".join(f"{split}={counts[split]}" for split in counts)
    )
    for sequence, dropped in unmatched.items():
        if dropped:
            print(
                f"Skipped {dropped} {sequence} image(s) without a Leica match "
                f"within {max_sync_delta_ms:g} ms."
            )
    return rows, counts


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_targets(rows, split):
    return np.asarray(
        [[r["x"], r["y"], r["z"]] for r in rows if r["split"] == split],
        dtype=np.float32,
    )


def _make_loader(dataset, batch_size, shuffle, workers, device):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )


def _run_epoch(model, loader, optimizer, criterion, device, coord_mean, coord_std):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        normalized_targets = (targets - coord_mean) / coord_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(images)
            loss = criterion(predictions, normalized_targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_count
        total_items += batch_count
    return total_loss / max(total_items, 1)


class _EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def update(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            return True, False
        self.epochs_without_improvement += 1
        return False, self.epochs_without_improvement >= self.patience


@torch.no_grad()
def _predict(model, loader, device, coord_mean, coord_std):
    model.eval()
    predictions, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        pred_norm = model(images)
        pred = pred_norm * coord_std + coord_mean
        predictions.append(pred.detach().cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def _measure_latency(model, dataset, device, coord_mean, coord_std, warmup=20, runs=100):
    model.eval()
    sample = dataset[0][0].unsqueeze(0).to(device)
    # Synchronize around timed GPU work; report model-forward latency, batch=1.
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def _hardware_info(device):
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    ram_gb = ""
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass
    return {
        "training_machine": socket.gethostname(),
        "gpu_model": gpu,
        "cpu_model": platform.processor() or platform.machine(),
        "ram_gb": ram_gb,
    }


def _append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_experiment(args, model_factory=create_model):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if not 0 <= args.freeze_pretrained_epochs <= args.epochs:
        raise ValueError(
            "--freeze-pretrained-epochs must be between 0 and --epochs."
        )
    set_seed(args.seed)
    device = choose_device(args.device)
    rows, counts = build_dataset_rows(
        args.data_root,
        args.val,
        args.test,
        args.max_sync_delta_ms,
        args.train_split,
        args.val_split,
        args.test_split,
    )

    train_ds = PrismImageDataset(rows, args.train_split, args.image_size)
    val_ds = PrismImageDataset(rows, args.val_split, args.image_size)
    train_loader = _make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = _make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    train_targets = _collect_targets(rows, args.train_split)
    mean_np, std_np = coordinate_statistics(train_targets)
    coord_mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    coord_std = torch.tensor(std_np, dtype=torch.float32, device=device)

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]
    results_csv = (
        Path(args.results_csv)
        if args.results_csv
        else _default_results_csv(args.model, args.test)
    )
    model = model_factory(args.model, checkpoint).to(device)
    if not hasattr(model, "set_pretrained_trainable"):
        raise TypeError(
            "Model factory must return a model with set_pretrained_trainable(bool)."
        )
    model.set_pretrained_trainable(True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.set_pretrained_trainable(args.freeze_pretrained_epochs == 0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    checkpoint_path = output_dir / f"{run_id}_best.pt"
    config_path = output_dir / f"{run_id}_config.json"

    config = vars(args).copy()
    config.update({
        "resolved_checkpoint": checkpoint,
        "device_used": str(device),
        "coordinate_mean_train_only": mean_np.tolist(),
        "coordinate_std_train_only": std_np.tolist(),
        "loss_function": "SmoothL1Loss",
        "optimizer": "AdamW",
        "pretrained_layers_frozen_epochs": args.freeze_pretrained_epochs,
        "dataset_root": str(Path(args.data_root).expanduser().resolve()),
        "validation_sequences": args.val,
        "test_sequences": args.test,
        "max_sync_delta_ms": args.max_sync_delta_ms,
        "sequence_split_sample_counts": counts,
        "results_csv": str(results_csv),
    })
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_training = time.perf_counter()
    early_stopping = _EarlyStopping(args.early_stopping_patience)
    best_epoch = 0
    epochs_completed = 0
    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        if epoch == args.freeze_pretrained_epochs + 1:
            model.set_pretrained_trainable(True)
            print("Unfroze pretrained layers; training the full model.")
        train_loss = _run_epoch(
            model, train_loader, optimizer, criterion, device, coord_mean, coord_std
        )
        val_loss = _run_epoch(
            model, val_loader, None, criterion, device, coord_mean, coord_std
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )
        improved, should_stop = early_stopping.update(val_loss)
        if improved:
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "checkpoint": checkpoint,
                "coordinate_mean": mean_np.tolist(),
                "coordinate_std": std_np.tolist(),
                "config": config,
            }, checkpoint_path)
        if should_stop:
            print(
                f"Early stopping after {epoch} epochs: validation loss did not "
                f"improve for {args.early_stopping_patience} consecutive epochs. "
                f"Best epoch: {best_epoch} (val_loss={early_stopping.best_loss:.6f})."
            )
            break

    training_time = time.perf_counter() - start_training

    # Reload the best validation checkpoint before touching the held-out test split.
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])

    # The test Dataset/loader is created only after training and checkpoint selection.
    test_ds = PrismImageDataset(rows, args.test_split, args.image_size)
    test_loader = _make_loader(test_ds, args.batch_size, False, args.num_workers, device)
    pred, target = _predict(model, test_loader, device, coord_mean, coord_std)
    metrics = localization_metrics(pred, target)
    latency_ms = _measure_latency(model, test_ds, device, coord_mean, coord_std)
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    else:
        peak_alloc = peak_reserved = ""

    row = {
        "experiment_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "model_name": args.model,
        "model_variant": checkpoint,
        "checkpoint_version": checkpoint,
        **_hardware_info(device),
        "epochs": epochs_completed,
        "best_epoch": best_epoch,
        "pretrained_layers_frozen_epochs": args.freeze_pretrained_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "loss_function": "SmoothL1Loss",
        "image_size": args.image_size,
        "random_seed": args.seed,
        "dataset_name": "MMAUD",
        "dataset_version": args.dataset_version,
        "modality": "RGB",
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "coordinate_frame": args.coordinate_frame,
        "coordinate_unit": args.coordinate_unit,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_time_seconds": round(training_time, 4),
        "inference_latency_ms_per_frame": round(latency_ms, 4),
        "inference_fps": round(1000.0 / latency_ms, 4),
        "peak_gpu_memory_allocated_gb": round(peak_alloc, 4) if peak_alloc != "" else "",
        "peak_gpu_memory_reserved_gb": round(peak_reserved, 4) if peak_reserved != "" else "",
        **metrics,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "notes": args.notes,
    }
    _append_csv(results_csv, row)
    print("\nHeld-out test metrics (meters):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nCSV row appended to: {results_csv.resolve()}")
    print(f"Best checkpoint: {checkpoint_path.resolve()}")
    return row


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
