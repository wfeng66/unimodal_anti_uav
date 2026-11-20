import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from mmaud3d.data import PrismImageDataset
from mmaud3d.metrics import coordinate_statistics, localization_metrics
from mmaud3d.models import (
    DinoDroneRegressor,
    TransformerRegressor,
    YoloPyramidRegressor,
    create_model,
)
from tune_hyperparameters import (
    _summary_filename,
    parse_args as parse_tuning_args,
)
from train import (
    CSV_FIELDS,
    _EarlyStopping,
    _append_csv,
    _default_results_csv,
    _resolve_data_root,
    build_dataset_rows,
    parse_args,
)


def test_windows_drive_path_is_translated_for_wsl():
    assert _resolve_data_root(
        r"I:\Data\Anti-UAV\MMAUD\v1", windows_host=False
    ) == Path("/mnt/i/Data/Anti-UAV/MMAUD/v1")


def test_default_results_filename_includes_camera_mode():
    timestamp = datetime(2026, 11, 10, tzinfo=timezone.utc)
    assert _default_results_csv("yolo", "Mavic3", "cat", timestamp) == Path(
        "results/yolo_mavic3_cat_11_2026.csv"
    )
    assert _default_results_csv(
        "dino-drone", "Avata,Mavic3", "cmb", timestamp
    ) == Path("results/dino-drone_avata_mavic3_cmb_11_2026.csv")


def test_tuning_summary_filename_includes_camera_mode():
    timestamp = datetime(2026, 11, 10, tzinfo=timezone.utc)

    assert _summary_filename("detr", "Mavic3", "mean", timestamp) == (
        "detr_mavic3_mean_11_2026.csv"
    )


def test_data_path_argument_and_data_root_alias():
    common = ["--model", "yolo", "--val", "avata", "--test", "m300"]
    assert parse_args(common + ["--data_path", "/datasets/mmaud"]).data_root == (
        "/datasets/mmaud"
    )
    assert parse_args(common + ["--data-root", "/datasets/other"]).data_root == (
        "/datasets/other"
    )
    assert parse_args(common).early_stopping_patience == 10
    assert parse_args(
        ["--model", "dino-drone", "--val", "avata", "--test", "m300"]
    ).model == "dino-drone"
    assert parse_tuning_args(
        [
            "--model", "dino-drone", "--data-root", "/data",
            "--val", "avata", "--test", "m300", "--2cam", "cmb",
        ]
    ).two_cam == "cmb"
    assert parse_args(
        ["--model", "yolo", "--val", "avata", "--test", "m300"]
    ).two_cam == "cat"
    assert parse_args(
        ["--model", "yolo", "--2cam", "mean", "--val", "avata", "--test", "m300"]
    ).two_cam == "mean"
    assert parse_args(
        ["--model", "yolo", "--2cam", "cmb", "--val", "avata", "--test", "m300"]
    ).two_cam == "cmb"


def test_early_stopping_waits_for_patience_and_resets_on_improvement():
    stopping = _EarlyStopping(patience=2)

    assert stopping.update(0.5) == (True, False)
    assert stopping.update(0.4) == (True, False)
    assert stopping.update(0.41) == (False, False)
    assert stopping.update(0.39) == (True, False)
    assert stopping.update(0.391) == (False, False)
    assert stopping.update(0.392) == (False, True)


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


def test_two_camera_dataset_modes_split_and_letterbox_views(tmp_path):
    image_path = tmp_path / "combined.png"
    combined = Image.new("RGB", (128, 48))
    combined.paste(Image.new("RGB", (64, 48), (255, 0, 0)), (0, 0))
    combined.paste(Image.new("RGB", (64, 48), (0, 255, 0)), (64, 0))
    combined.save(image_path)
    rows = [{
        "image_path": str(image_path),
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "split": "train",
    }]

    cat_views, target = PrismImageDataset(rows, "train", 28, "cat")[0]
    assert cat_views.shape == (2, 3, 28, 28)
    assert cat_views[0, 0, 10, 10] == 1
    assert cat_views[1, 1, 10, 10] == 1
    assert torch.allclose(
        cat_views[0, :, 0, 0],
        torch.tensor([124, 116, 104], dtype=torch.float32) / 255,
    )
    assert torch.equal(target, torch.tensor([1.0, 2.0, 3.0]))

    independent = PrismImageDataset(rows, "train", 28, "ind")
    assert len(independent) == 2
    assert independent[0][0][0, 10, 10] == 1
    assert independent[1][0][1, 10, 10] == 1
    assert PrismImageDataset(rows, "train", 28, "left")[0][0].shape == (3, 28, 28)
    assert PrismImageDataset(rows, "train", 28, "right")[0][0][1, 10, 10] == 1
    assert PrismImageDataset(rows, "train", 28, "mean")[0][0].shape == (2, 3, 28, 28)

    combined_as_one, _ = PrismImageDataset(rows, "train", 28, "cmb")[0]
    original = np.array(
        combined.resize((28, 28), Image.Resampling.BILINEAR),
        dtype=np.float32,
        copy=True,
    ) / 255.0
    expected = torch.from_numpy(original).permute(2, 0, 1)
    assert combined_as_one.shape == (3, 28, 28)
    assert torch.equal(combined_as_one, expected)


def test_coordinate_metrics_and_zero_variance():
    mean, std = coordinate_statistics([[2, 4, 6], [2, 6, 8]])
    assert np.allclose(mean, [2, 5, 7])
    assert np.allclose(std, [1, 1, 1])

    metrics = localization_metrics([[1, 2, 2]], [[1, 2, 3]])
    assert metrics["mean_euclidean_error_m"] == 1
    assert metrics["mae_z_m"] == 1
    assert metrics["rmse_3d_m"] == 1


def test_results_csv_migrates_old_header_before_appending_two_camera_mode(tmp_path):
    results_path = tmp_path / "results.csv"
    old_fields = [field for field in CSV_FIELDS if field != "two_camera_mode"]
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({"experiment_id": "previous-run", "model_name": "yolo"})

    _append_csv(results_path, {
        "experiment_id": "new-run",
        "model_name": "yolo",
        "two_camera_mode": "cat",
    })

    with results_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == CSV_FIELDS
    assert rows[0]["experiment_id"] == "previous-run"
    assert rows[0]["two_camera_mode"] == ""
    assert rows[1]["two_camera_mode"] == "cat"


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


@pytest.mark.parametrize("mode", ["cat", "mean"])
def test_yolo_regressor_two_camera_modes(mode):
    model = YoloPyramidRegressor(_FakeDetectionModel(), two_cam=mode)

    prediction = model(torch.rand(2, 2, 3, 32, 32))

    assert prediction.shape == (2, 3)
    assert model.regression_head[0].in_features == (16 if mode == "cat" else 8)


class _FakeDinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            patch_size=14,
            num_hidden_layers=12,
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.requested_hidden_states = False

    def forward(self, pixel_values, output_hidden_states=False):
        self.requested_hidden_states = output_hidden_states
        batch, _, height, width = pixel_values.shape
        sequence_length = (height // 14) * (width // 14) + 1
        hidden_states = tuple(
            torch.full(
                (batch, sequence_length, 8),
                float(index),
                device=pixel_values.device,
            ) * self.scale
            for index in range(13)
        )
        return SimpleNamespace(
            hidden_states=hidden_states,
            last_hidden_state=hidden_states[-1],
        )


def test_dino_drone_uses_pretrained_multiscale_features_and_returns_coordinates():
    pretrained = _FakeDinoBackbone()
    model = DinoDroneRegressor(pretrained, hidden_size=8, patch_size=14)

    prediction = model(torch.rand(2, 3, 56, 56))

    assert prediction.shape == (2, 3)
    assert pretrained.requested_hidden_states
    assert all(parameter.requires_grad for parameter in model.feature_pyramid.parameters())

    model.set_pretrained_trainable(False)
    model.train()
    assert not pretrained.training
    assert all(not parameter.requires_grad for parameter in pretrained.parameters())
    assert all(parameter.requires_grad for parameter in model.regression_head.parameters())


def test_dino_drone_uses_reported_intermediate_transformer_blocks():
    model = DinoDroneRegressor(_FakeDinoBackbone(), hidden_size=8, patch_size=14)
    captured_features = []

    def capture_inputs(module, inputs):
        assert module is model.feature_pyramid
        captured_features.append([
            feature[:, :, 0, 0].mean().item()
            for feature in inputs[0]
        ])

    handle = model.feature_pyramid.register_forward_pre_hook(capture_inputs)
    model(torch.rand(2, 3, 56, 56))
    handle.remove()

    assert captured_features == [[3.0, 6.0, 9.0, 12.0]]


def test_dino_drone_requires_patch_aligned_image_size():
    model = DinoDroneRegressor(_FakeDinoBackbone(), hidden_size=8, patch_size=14)

    with pytest.raises(ValueError, match="divisible"):
        model(torch.rand(2, 3, 55, 56))


@pytest.mark.parametrize("mode", ["cat", "mean"])
def test_transformer_regressor_two_camera_modes(mode):
    pretrained = _FakeDinoBackbone()
    model = TransformerRegressor(pretrained, hidden_size=8, two_cam=mode)
    images = torch.rand(2, 2, 3, 28, 28)

    prediction = model(images)

    assert prediction.shape == (2, 3)
    assert model.regression_head[0].in_features == (16 if mode == "cat" else 8)


def test_transformer_regressor_combined_mode_uses_single_image_path():
    model = TransformerRegressor(_FakeDinoBackbone(), hidden_size=8, two_cam="cmb")

    prediction = model(torch.rand(2, 3, 28, 28))

    assert prediction.shape == (2, 3)
    assert model.regression_head[0].in_features == 8


@pytest.mark.parametrize("mode", ["cat", "mean"])
def test_dino_drone_two_camera_modes(mode):
    model = DinoDroneRegressor(
        _FakeDinoBackbone(), hidden_size=8, patch_size=14, two_cam=mode
    )

    prediction = model(torch.rand(2, 2, 3, 56, 56))

    assert prediction.shape == (2, 3)
    assert model.regression_head[0].in_features == (1536 if mode == "cat" else 768)


def test_dino_drone_factory_loads_pretrained_checkpoint(monkeypatch):
    pretrained = _FakeDinoBackbone()
    loaded_checkpoints = []
    transformers = ModuleType("transformers")

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(checkpoint):
            loaded_checkpoints.append(checkpoint)
            return pretrained

    setattr(transformers, "AutoModel", FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model = create_model("dino-drone", "facebook/dinov2-small", "cat")

    assert loaded_checkpoints == ["facebook/dinov2-small"]
    assert model.pretrained is pretrained
    assert model.regression_head[0].in_features == 1536
