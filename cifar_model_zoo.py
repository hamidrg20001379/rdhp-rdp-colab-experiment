#!/usr/bin/env python3
"""CIFAR-10 model definitions used by the cross-model pruning benchmark.

The ResNet-32 definition is loaded from the pinned thesis source. VGG-16 and
DenseNet-121 use CIFAR-compatible adaptations of torchvision architectures;
their metadata is recorded by the runner so these are not confused with an
unverified author implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch.nn as nn
from torchvision.models import densenet121, vgg16


MODEL_NAMES = ("resnet32", "vgg16", "densenet121")


class CifarVGG16(nn.Module):
    """VGG-16-style CIFAR model with a compact 512-to-10 classifier."""

    _CONFIG = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M")

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for value in self._CONFIG:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.extend([
                nn.Conv2d(channels, value, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
            channels = value
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, inputs):
        values = self.avgpool(self.features(inputs))
        return self.classifier(values.flatten(1))


def _load_resnet32(thesis_root: Path) -> nn.Module:
    source = thesis_root / "working" / "experiments" / "kaggle_sources" / "resnet_cifar.py"
    if not source.exists():
        raise FileNotFoundError(
            f"Pinned ResNet source not found at {source}; pass --thesis-root "
            "pointing to the checked-out my-thesis repository."
        )
    spec = importlib.util.spec_from_file_location("pinned_resnet_cifar", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet32_cifar()


def load_model(name: str, *, thesis_root: Path | None = None) -> nn.Module:
    """Create an uninitialized 10-class model for ``name``."""
    if name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {name!r}")
    if name == "resnet32":
        if thesis_root is None:
            raise ValueError("thesis_root is required for the pinned ResNet-32 source")
        return _load_resnet32(thesis_root)
    if name == "vgg16":
        return CifarVGG16(num_classes=10)
    model = densenet121(weights=None)
    # CIFAR images do not need ImageNet's 7x7 stride-2 stem or max pool.
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model


def model_description(name: str) -> str:
    descriptions = {
        "resnet32": "pinned RD_PRUNE CIFAR ResNet-32 source",
        "vgg16": "CIFAR-compatible VGG-16-style model with adaptive 1x1 pooling",
        "densenet121": "torchvision DenseNet-121 with CIFAR stem and 10-class head",
    }
    return descriptions[name]
