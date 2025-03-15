import os

import numpy as np
import torch

from src.dataset import (
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_medmnist_dataset_with_single_label,
    get_mnist_dataset,
)
from src.model import get_resnet18
from src.verify import get_acc

ENV = "ayame"
DEVICE = "cuda:0"

# OUTPUT_DIR = "output"
OUTPUT_DIR = "/nas/keito/ML_diff_experiment/output2"

DATAFLAGS = [
    "mnist",
    "cifar10",
    "cifar100",
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

# NOW = now()
NOW = "2025-03-15-02-13-54"


def is_exist(path: str):
    if os.path.isfile(path):
        print(f"o | {path}")
    else:
        print(f"x | {path}")


def check():
    for data_flag in DATAFLAGS:
        print(f"==================== {data_flag} ====================")

        is_exist(f"{OUTPUT_DIR}/{data_flag}/default/resnet18_{NOW}_{ENV}.pth")

        for rate in np.arange(0.05, 0.55, 0.05).astype(float):
            is_exist(
                f"{OUTPUT_DIR}/{data_flag}/rate_{rate:.2f}/resnet18_{NOW}_{ENV}.pth"
            )
            is_exist(
                f"{OUTPUT_DIR}/{data_flag}/rate_{rate:.2f}/indices_{NOW}_{ENV}.pkl"
            )

        for target_class in range(0):
            for rate in np.arange(0.5, 1.05, 0.05).astype(float):
                is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/resnet18_{NOW}_{ENV}.pth"
                )
                is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/indices_{NOW}_{ENV}.pkl"
                )


def main(batch_size=128, device="cuda:0"):
    for data_flag in DATAFLAGS:
        if data_flag == "mnist":
            num_channels = 1
            num_classes = 10
            train_dataset, val_dataset, test_dataset = get_mnist_dataset()
        elif data_flag == "cifar10":
            num_channels = 3
            num_classes = 10
            train_dataset, val_dataset, test_dataset = get_cifar10_dataset()
        elif data_flag == "cifar100":
            num_channels = 3
            num_classes = 100
            train_dataset, val_dataset, test_dataset = get_cifar100_dataset()
        else:
            (
                train_dataset,
                val_dataset,
                test_dataset,
                task,
                num_channels,
                num_classes,
            ) = get_medmnist_dataset_with_single_label(data_flag)

        default_model = get_resnet18(num_channels=num_channels, num_classes=num_classes)
        default_model.load_state_dict(
            torch.load(f"{OUTPUT_DIR}/{data_flag}/default/resnet18_{NOW}_{ENV}.pth")
        )
        default_acc = get_acc(
            default_model,
            train_dataset,
            num_classes,
            batch_size=batch_size,
            device=device,
        )


check()
