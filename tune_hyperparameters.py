#!/usr/bin/env python3
"""Run a reproducible random search over train.py hyperparameters."""

import argparse
import csv
import itertools
import math
import random
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_DIR / "train.py"
VAL_LOSS_PATTERN = re.compile(
    r"Epoch\s+(\d+)/\d+:\s+train_loss=[^,]+,\s+val_loss=([0-9.eE+-]+)"
)
SUMMARY_FIELDS = (
    "trial",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "freeze_pretrained_epochs",
    "best_epoch",
    "best_val_loss",
    "log_path",
)


def _csv_values(value, converter, option):
    try:
        values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be comma-separated values.") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option} must contain at least one value.")
    return values


def _positive_floats(value):
    values = _csv_values(value, float, "--learning-rates")
    if any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Learning rates must be finite and positive.")
    return values


def _nonnegative_floats(value):
    values = _csv_values(value, float, "--weight-decays")
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("Weight decays must be finite and non-negative.")
    return values


def _positive_ints(value):
    values = _csv_values(value, int, "--batch-sizes")
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Batch sizes must be positive.")
    return values


def _nonnegative_ints(value):
    values = _csv_values(value, int, "--freeze-epochs")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Freeze epochs must be non-negative.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Random-search train.py settings, selecting by minimum validation loss. "
            "Do not use the reported test metrics to choose hyperparameters."
        )
    )
    parser.add_argument(
        "--model", choices=("yolo", "detr", "dinov2", "dino-drone"), default="yolo"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val", required=True, help="Validation recording(s).")
    parser.add_argument(
        "--test",
        required=True,
        help=(
            "train.py requires a test recording. Prefer an internal recording here; "
            "keep the final held-out test set untouched."
        ),
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rates", type=_positive_floats, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--weight-decays", type=_nonnegative_floats, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--batch-sizes", type=_positive_ints, default=[8, 16])
    parser.add_argument("--freeze-epochs", type=_nonnegative_ints, default=[0, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="runs/hyperparameter_search")
    args = parser.parse_args(argv)

    if args.trials < 1 or args.epochs < 1 or args.image_size < 1 or args.num_workers < 0:
        parser.error("trials, epochs, and image-size must be positive; num-workers cannot be negative.")
    if any(value > args.epochs for value in args.freeze_epochs):
        parser.error("Every --freeze-epochs value must be at most --epochs.")
    return args


def _write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run_trial(args, trial_number, values, output_root):
    learning_rate, weight_decay, batch_size, freeze_epochs = values
    trial_dir = output_root / f"trial_{trial_number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model", args.model,
        "--data-root", args.data_root,
        "--val", args.val,
        "--test", args.test,
        "--epochs", str(args.epochs),
        "--learning-rate", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--batch-size", str(batch_size),
        "--freeze-pretrained-epochs", str(freeze_epochs),
        "--seed", str(args.seed),
        "--image-size", str(args.image_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--output-dir", str(trial_dir / "checkpoints"),
    ]

    print(f"\n=== Trial {trial_number}: {values} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()

    if return_code:
        raise RuntimeError(
            f"Trial {trial_number} failed with exit code {return_code}; see {log_path}."
        )

    losses = [
        (int(match.group(1)), float(match.group(2)))
        for match in VAL_LOSS_PATTERN.finditer(log_path.read_text(encoding="utf-8"))
    ]
    if not losses:
        raise RuntimeError(
            f"Trial {trial_number} completed without any parsed validation losses; "
            f"see {log_path}."
        )
    best_epoch, best_val_loss = min(losses, key=lambda item: item[1])
    return {
        "trial": trial_number,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "freeze_pretrained_epochs": freeze_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "log_path": str(log_path),
    }


def main(argv=None):
    args = parse_args(argv)
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "tuning_summary.csv"

    candidates = list(itertools.product(
        args.learning_rates,
        args.weight_decays,
        args.batch_sizes,
        args.freeze_epochs,
    ))
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[:min(args.trials, len(candidates))]
    if len(selected) < args.trials:
        print(f"Only {len(selected)} unique hyperparameter combinations are available.")

    results = []
    for trial_number, values in enumerate(selected, start=1):
        result = _run_trial(
            args, trial_number, values, output_root
        )
        results.append(result)
        _write_summary(summary_path, results)

    results.sort(key=lambda result: result["best_val_loss"])
    print("\nRanked by best validation loss (lower is better):")
    for result in results:
        print(
            f"trial {result['trial']:03d}: val_loss={result['best_val_loss']:.6f}, "
            f"lr={result['learning_rate']}, wd={result['weight_decay']}, "
            f"batch={result['batch_size']}, "
            f"freeze={result['freeze_pretrained_epochs']} "
            f"(epoch {result['best_epoch']})"
        )
    print(f"\nSummary: {summary_path}")
    print("Use validation loss—not the test metrics in train_results.csv—to select a trial.")


if __name__ == "__main__":
    main()
