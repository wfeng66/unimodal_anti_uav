from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PrismImageDataset(Dataset):
    def __init__(self, rows, split, image_size):
        self.rows = [row for row in rows if row["split"] == split]
        self.image_size = image_size
        if not self.rows:
            raise ValueError(f"No samples found for split {split!r}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        target = torch.tensor([row["x"], row["y"], row["z"]], dtype=torch.float32)
        return image_tensor, target
