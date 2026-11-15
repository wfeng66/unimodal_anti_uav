import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mmaud3d.metrics import coordinate_statistics, localization_metrics
from mmaud3d.models import YoloPyramidRegressor
from train import (
    _default_results_csv,
    _resolve_data_root,
    build_dataset_rows,
    parse_args,
)


def test_windows_drive_path_is_translated_for_wsl():
    assert _resolve_data_root(
        r"I:\Data\Anti-UAV\MMAUD\v1", windows_host=False
    ) == Path("/mnt/i/Data/Anti-UAV/MMAUD/v1")


def test_default_results_filename_uses_method_test_and_month_year():
    timestamp = datetime(2026, 11, 10, tzinfo=timezone.utc)
    assert _default_results_csv("yolo", "Mavic3", timestamp) == Path(
        "results/yolo_mavic3_11_2026.csv"
    )


def test_data_path_argument_and_data_root_alias():
    common = ["--model", "yolo", "--val", "avata", "--test", "m300"]
    assert parse_args(common + ["--data_path", "/datasets/mmaud"]).data_root == (
        "/datasets/mmaud"
    )
    assert parse_args(common + ["--data-root", "/datasets/other"]).data_root == (
        "/datasets/other"
    )


def _add_sequence(root, name, timestamps, points):
    images = root / name / "usb_cam_image_raw" / "decoded"
    images.mkdir(parents=True)
    gt = root / f"2023-08-24-11-47-30_{name.lower()}" / "leica_point_relative"
    decoded = gt / "decoded"
    decoded.mkdir(parents=True)
    index_path = gt / "timestamps.csv"
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("timestamp_ns", "decoded_file"))
        writer.writeheader()
        for i, (timestamp, point) in enumerate(zip(timestamps, points)):
            image_timestamp = timestamp
            (images / f"{i:08d}_{image_timestamp}.png").touch()
            annotation_name = f"{i:08d}_{timestamp}.json"
            (decoded / annotation_name).write_text(
                json.dumps({"annotation": {"point": point}}), encoding="utf-8"
            )
            writer.writerow({
                "timestamp_ns": timestamp,
                "decoded_file": f"leica_point_relative/decoded/{annotation_name}",
            })


def test_build_dataset_rows_pairs_relative_leica_and_splits(tmp_path):
    base = 1_692_848_852_000_000_000
    _add_sequence(tmp_path, "Avata", [base], [[1.0, 2.0, 3.0]])
    _add_sequence(tmp_path, "M300", [base], [[4.0, 5.0, 6.0]])
    _add_sequence(tmp_path, "Mavic3", [base], [[7.0, 8.0, 9.0]])
    image_dir = tmp_path / "Mavic3" / "usb_cam_image_raw" / "decoded"
    first_image = image_dir / f"00000000_{base}.png"
    first_image.rename(image_dir / f"00000000_{base + 100_000_000}.png")
    (image_dir / f"00000001_{base + 100_000_001}.png").touch()

    rows, counts = build_dataset_rows(tmp_path, "avata", "m300", 100.0)

    assert counts == {"train": 1, "val": 1, "test": 1}
    assert len(rows) == 3
    avata = next(row for row in rows if row["sequence"] == "Avata")
    assert (avata["x"], avata["y"], avata["z"]) == (1.0, 2.0, 3.0)
    assert avata["split"] == "val"


def test_coordinate_metrics_and_zero_variance():
    mean, std = coordinate_statistics([[2, 4, 6], [2, 6, 8]])
    assert np.allclose(mean, [2, 5, 7])
    assert np.allclose(std, [1, 1, 1])

    metrics = localization_metrics([[1, 2, 2]], [[1, 2, 3]])
    assert metrics["mean_euclidean_error_m"] == 1
    assert metrics["mae_z_m"] == 1
    assert metrics["rmse_3d_m"] == 1


class _FakeDetect(nn.Module):
    i = 1
    f = [-1]

    def forward(self, features):
        return features


class _FakeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 8, 1), _FakeDetect())

    def forward(self, images):
        features = self.model[0](images)
        return self.model[-1]([features])


def test_yolo_pyramid_regressor_freeze_then_unfreeze():
    model = YoloPyramidRegressor(_FakeDetectionModel())
    model.set_pretrained_trainable(False)
    model.train()
    assert not model.pretrained.training
    assert all(not parameter.requires_grad for parameter in model.pretrained.parameters())
    assert all(parameter.requires_grad for parameter in model.regression_head.parameters())
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 3)

    model.set_pretrained_trainable(True)
    model.train()
    assert model.pretrained.training
    assert all(parameter.requires_grad for parameter in model.pretrained.parameters())
