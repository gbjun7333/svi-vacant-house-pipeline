"""ResNet-50 classifier with one output logit, vacant against non-vacant (Section 3.4)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def create_model(pretrained: bool = True) -> nn.Module:
    """ResNet-50 with ImageNet weights. pretrained=False when a checkpoint is loaded afterwards."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def widen_conv1(model: nn.Module) -> nn.Module:
    """First convolution with four input channels (RGB + mask).

    The mask filters start at zero, so the network equals the RGB model before training.
    """
    old = model.conv1
    new = torch.nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                          bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.conv1 = new
    return model
