import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CHECKPOINTS = {
    "yolo": "yolo26n.pt",
    "detr": "facebook/detr-resnet-50",
    "dinov2": "facebook/dinov2-base",
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


def create_model(model_name, checkpoint):
    if model_name == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO training requires ultralytics; install requirements.txt."
            ) from exc
        return YoloPyramidRegressor(YOLO(checkpoint).model)

    if model_name in ("detr", "dinov2"):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DETR and DINOv2 training require transformers; install requirements.txt."
            ) from exc
        pretrained = AutoModel.from_pretrained(checkpoint)
        config = pretrained.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Cannot determine hidden size for checkpoint {checkpoint!r}.")
        return TransformerRegressor(pretrained, hidden_size)

    raise ValueError(f"Unsupported model name: {model_name!r}")
