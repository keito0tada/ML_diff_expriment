import torch.nn as nn
from torchvision.models import resnet18, vit_b_16


def get_resnet18(num_channels, num_classes):
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    return model


def get_vit_b_16(num_channels, num_classes):
    model = vit_b_16(pretrained=False, num_classes=num_classes)
    return model


def get_model(arch: str, num_channels: int, num_classes: int):
    if arch == "resnet18":
        return get_resnet18(num_channels=num_channels, num_classes=num_classes)
    else:
        raise ValueError
