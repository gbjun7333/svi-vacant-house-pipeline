"""Classifier of the classification experiment (Section 3.4).

ResNet-50 with ImageNet weights and one output logit (vacant against non-vacant).
For the conditions that carry a mask, the first convolution is widened to four
input channels: the RGB filters keep the pretrained weights and the filters of the
mask channel start at zero, so before training the network equals the
three-channel model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def create_model(pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a single-logit head. ``pretrained=False`` skips the ImageNet
    weights, for a model whose weights are loaded from a checkpoint afterwards."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def widen_conv1(model: nn.Module) -> nn.Module:
    """Replace the first convolution by one with four input channels (RGB + mask)."""
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
