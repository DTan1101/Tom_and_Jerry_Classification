from typing import Tuple

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "efficientnet_b0": models.efficientnet_b0,
}


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(SUPPORTED_MODELS)}")

    model_fn = SUPPORTED_MODELS[model_name]
    model = model_fn(weights="DEFAULT" if pretrained else None)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
