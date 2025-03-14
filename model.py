import torch.nn as nn
from torchvision.models import resnet18, vit_b_16


def get_resnet18(num_channels, num_classes):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    return model


def get_vit_b_16(num_channels, num_classes):
    model = vit_b_16(pretrained=False, num_classes=num_classes)
    return model
