import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CHECKPOINTS = {
    "yolo": "yolo26n.pt",
    "detr": "facebook/detr-resnet-50",
    "dinov2": "facebook/dinov2-base",
    "dino-drone": "facebook/dinov2-small",
}


class _FeaturePassthrough(nn.Module):
    def __init__(self, detection_head):
        super().__init__()
        self.i = detection_head.i
        self.f = detection_head.f
        self.type = "FeaturePassthrough"
        self.np = 0

    def forward(self, features):
        return features


class YoloPyramidRegressor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        self._pretrained_trainable = True
        model_layers = getattr(self.pretrained, "model", None)
        if model_layers is None or len(model_layers) < 2:
            raise ValueError("The YOLO checkpoint has no expected detection model layers.")
        detection_head = model_layers[-1]
        if "detect" not in type(detection_head).__name__.lower():
            raise ValueError(
                "YOLO checkpoint must be a detection model (for example yolo26n.pt), "
                "not a classification checkpoint."
            )
        model_layers[-1] = _FeaturePassthrough(detection_head)

        was_training = self.pretrained.training
        self.pretrained.eval()
        try:
            with torch.no_grad():
                features = self.pretrained(torch.zeros(1, 3, 256, 256))
        finally:
            self.pretrained.train(was_training)
        if not isinstance(features, (list, tuple)) or not features:
            raise TypeError("YOLO detection model did not return pyramid feature maps.")
        channels = []
        for feature in features:
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise TypeError("YOLO pyramid outputs must be 4D feature tensors.")
            channels.append(feature.shape[1])

        self.regression_head = nn.Sequential(
            nn.Linear(sum(channels), 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

    def set_pretrained_trainable(self, trainable):
        self._pretrained_trainable = trainable
        for parameter in self.pretrained.parameters():
            parameter.requires_grad = trainable
        for parameter in self.regression_head.parameters():
            parameter.requires_grad = True
        if not trainable:
            self.pretrained.eval()
        elif self.training:
            self.pretrained.train()

    def train(self, mode=True):
        super().train(mode)
        if not self._pretrained_trainable:
            self.pretrained.eval()
        return self

    def forward(self, images):
        features = self.pretrained(images)
        pooled = [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in features]
        return self.regression_head(torch.cat(pooled, dim=1))


class TransformerRegressor(nn.Module):
    def __init__(self, pretrained_model, hidden_size):
        super().__init__()
        self.pretrained = pretrained_model
        self._pretrained_trainable = True
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def set_pretrained_trainable(self, trainable):
        self._pretrained_trainable = trainable
        for parameter in self.pretrained.parameters():
            parameter.requires_grad = trainable
        for parameter in self.regression_head.parameters():
            parameter.requires_grad = True
        if not trainable:
            self.pretrained.eval()
        elif self.training:
            self.pretrained.train()

    def train(self, mode=True):
        super().train(mode)
        if not self._pretrained_trainable:
            self.pretrained.eval()
        return self

    def forward(self, images):
        images = (images - self.image_mean) / self.image_std
        outputs = self.pretrained(pixel_values=images)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(outputs, "decoder_last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("Pretrained transformer returned no hidden states.")
        return self.regression_head(hidden[:, 0])


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DinoDroneFeaturePyramid(nn.Module):
    def __init__(self, hidden_size, feature_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.lateral_projections = nn.ModuleList(
            _ConvBNAct(hidden_size, feature_size, kernel_size=1)
            for _ in range(4)
        )
        self.merge = nn.Sequential(
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
        )
        self.p3 = nn.Sequential(
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
        )
        self.p4 = nn.Sequential(
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
        )
        self.p5 = nn.Sequential(
            _ConvBNAct(feature_size, feature_size, kernel_size=3, stride=2, padding=1),
            _ConvBNAct(feature_size, feature_size, kernel_size=3, padding=1),
        )

    def forward(self, features):
        if len(features) != len(self.lateral_projections):
            raise ValueError("DINOFPN requires exactly four DINOv2 feature maps.")
        projected = [
            projection(feature)
            for projection, feature in zip(self.lateral_projections, features)
        ]
        merged = self.merge(torch.stack(projected, dim=0).mean(dim=0))
        p3_input = F.interpolate(merged, scale_factor=2, mode="nearest")
        return self.p3(p3_input), self.p4(merged), self.p5(merged)


class DinoDroneRegressor(TransformerRegressor):
    # These are zero-based DINOv2 transformer-block indices used by DINO-Drone.
    FEATURE_BLOCK_INDICES = (2, 5, 8, 11)

    def __init__(self, pretrained_model, hidden_size, patch_size):
        super().__init__(pretrained_model, hidden_size)
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            self.patch_size = tuple(int(size) for size in patch_size)
        else:
            raise ValueError(f"Unsupported DINOv2 patch size: {patch_size!r}.")
        if any(size < 1 for size in self.patch_size):
            raise ValueError(f"Unsupported DINOv2 patch size: {patch_size!r}.")

        self.hidden_size = hidden_size
        self.feature_pyramid = DinoDroneFeaturePyramid(hidden_size)
        self.regression_head = nn.Sequential(
            nn.Linear(3 * self.feature_pyramid.feature_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
        )

    def _feature_map(self, hidden, grid_height, grid_width):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise TypeError("DINOv2 hidden states must have shape (batch, tokens, channels).")
        patch_count = grid_height * grid_width
        if hidden.shape[1] < patch_count:
            raise ValueError("DINOv2 hidden state has fewer tokens than image patches.")
        patches = hidden[:, -patch_count:, :]
        return patches.reshape(
            hidden.shape[0], grid_height, grid_width, self.hidden_size
        ).permute(0, 3, 1, 2).contiguous()

    def forward(self, images):
        patch_height, patch_width = self.patch_size
        height, width = images.shape[-2:]
        if height % patch_height or width % patch_width:
            raise ValueError(
                "DINO-Drone input height and width must be divisible by the "
                f"DINOv2 patch size {self.patch_size}."
            )

        images = (images - self.image_mean) / self.image_std
        outputs = self.pretrained(pixel_values=images, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Pretrained DINOv2 returned no hidden states.")
        final_hidden_state_index = max(self.FEATURE_BLOCK_INDICES) + 1
        if len(hidden_states) <= final_hidden_state_index:
            raise ValueError(
                "The DINOv2 checkpoint must expose hidden states through transformer "
                f"block {max(self.FEATURE_BLOCK_INDICES)}."
            )

        # Hugging Face hidden_states[0] is the embedding output; block i is at i + 1.
        feature_maps = [
            self._feature_map(
                hidden_states[block_index + 1],
                height // patch_height,
                width // patch_width,
            )
            for block_index in self.FEATURE_BLOCK_INDICES
        ]
        pyramid = self.feature_pyramid(feature_maps)
        pooled = [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in pyramid]
        return self.regression_head(torch.cat(pooled, dim=1))


def create_model(model_name, checkpoint):
    if model_name == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO training requires ultralytics; install requirements.txt."
            ) from exc
        return YoloPyramidRegressor(YOLO(checkpoint).model)

    if model_name in ("detr", "dinov2", "dino-drone"):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DETR, DINOv2, and DINO-Drone training require transformers; "
                "install requirements.txt."
            ) from exc
        pretrained = AutoModel.from_pretrained(checkpoint)
        config = pretrained.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Cannot determine hidden size for checkpoint {checkpoint!r}.")
        if model_name == "dino-drone":
            patch_size = getattr(config, "patch_size", None)
            if patch_size is None:
                raise ValueError(
                    f"Cannot determine DINOv2 patch size for checkpoint {checkpoint!r}."
                )
            return DinoDroneRegressor(pretrained, hidden_size, patch_size)
        return TransformerRegressor(pretrained, hidden_size)

    raise ValueError(f"Unsupported model name: {model_name!r}")
