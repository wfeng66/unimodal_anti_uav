from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class PrismImageDataset(Dataset):
    TWO_CAMERA_MODES = ("cat", "ind", "left", "right", "mean", "cmb")

    def __init__(self, rows, split, image_size, two_cam="cat"):
        self.rows = [row for row in rows if row["split"] == split]
        self.image_size = image_size
        if two_cam not in self.TWO_CAMERA_MODES:
            raise ValueError(f"Unsupported two-camera mode: {two_cam!r}.")
        self.two_cam = two_cam
        if two_cam == "ind":
            self.samples = [
                (row, camera_index)
                for row in self.rows
                for camera_index in (0, 1)
            ]
        else:
            self.samples = [(row, None) for row in self.rows]
        if not self.rows:
            raise ValueError(f"No samples found for split {split!r}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row, camera_index = self.samples[index]
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as source:
            combined = source.convert("RGB")
            if self.two_cam == "cmb":
                image = combined.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                image_tensor = torch.from_numpy(
                    np.array(image, dtype=np.float32, copy=True) / 255.0
                ).permute(2, 0, 1)
            else:
                if combined.width % 2:
                    raise ValueError(
                        f"Two-camera image width must be even to split at the midpoint: "
                        f"{image_path} has width {combined.width}."
                    )
                midpoint = combined.width // 2
                views = (
                    combined.crop((0, 0, midpoint, combined.height)),
                    combined.crop((midpoint, 0, combined.width, combined.height)),
                )
                views = tuple(
                    ImageOps.pad(
                        view,
                        (self.image_size, self.image_size),
                        method=Image.Resampling.BILINEAR,
                        color=(124, 116, 104),
                    )
                    for view in views
                )
                images = torch.stack([
                    torch.from_numpy(
                        np.array(view, dtype=np.float32, copy=True) / 255.0
                    ).permute(2, 0, 1)
                    for view in views
                ])
                if self.two_cam == "left":
                    image_tensor = images[0]
                elif self.two_cam == "right":
                    image_tensor = images[1]
                elif self.two_cam == "ind":
                    image_tensor = images[camera_index]
                else:
                    image_tensor = images
        target = torch.tensor([row["x"], row["y"], row["z"]], dtype=torch.float32)
        return image_tensor, target
