"""Model definitions: the original scratch CNN plus pretrained baselines.

ResNet9 reimplements the architecture from the original notebook so the two can
be compared under identical training conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def conv_block(
    in_channels: int, out_channels: int, pool: bool = False, pool_size: int = 4
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_size))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """The original notebook's architecture: a compact ResNet-style CNN.

    The aggressive MaxPool2d(4) is kept from the original. Pooling by 2 instead
    leaves conv3 and conv4 on 112x112 and 56x56 feature maps, which makes the
    network about an order of magnitude slower than ResNet18.

    Only the head differs: adaptive average pooling, so any input resolution
    works rather than just the one it was written for.
    """

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap an ImageNet 1000-way head for a `num_classes`-way one."""
    if hasattr(model, "fc"):  # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):  # EfficientNet / MobileNet family
        head = model.classifier
        if isinstance(head, nn.Sequential):
            last = head[-1]
            head[-1] = nn.Linear(last.in_features, num_classes)
        else:
            model.classifier = nn.Linear(head.in_features, num_classes)
    else:
        raise ValueError(f"Don't know how to replace the head of {type(model).__name__}")
    return model


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Construct a model by name. resnet9 ignores `pretrained`."""
    name = name.lower()

    if name == "resnet9":
        return ResNet9(num_classes)

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_head(models.resnet18(weights=weights), num_classes)

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return _replace_head(models.resnet50(weights=weights), num_classes)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_head(models.efficientnet_b0(weights=weights), num_classes)

    raise ValueError(f"Unknown model '{name}'")


def target_layer_for_gradcam(model: nn.Module, name: str) -> nn.Module:
    """The last convolutional stage: deep enough to be semantic, but before
    pooling destroys the spatial detail Grad-CAM needs."""
    name = name.lower()
    if name == "resnet9":
        return model.res2
    if name in {"resnet18", "resnet50"}:
        return model.layer4
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer registered for '{name}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
