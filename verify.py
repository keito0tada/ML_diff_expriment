import os
import pickle

import numpy as np
import torch
from torcheval.metrics.functional import multiclass_accuracy

from src.dataset import (
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_medmnist_dataset_with_single_label,
    get_mnist_dataset,
)
from src.logger import logger_regular
from src.model import get_resnet18
from src.verify import get_targets_and_outputs

ENV = "ayame"
DEVICE = "cuda:0"

# OUTPUT_DIR = "output"
OUTPUT_DIR = "/nas/keito/ML_diff_experiment/output4"

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
        return True
    else:
        print(f"x | {path}")
        return False


def check():
    all = 0
    exist = 0
    for data_flag in DATAFLAGS:
        print(f"==================== {data_flag} ====================")

        NOW = "2025-03-16-03-13-37"
        ENV = "ayame"
        all += 1
        exist += is_exist(f"{OUTPUT_DIR}/{data_flag}/default/resnet18_{NOW}_{ENV}.pth")

        for rate in np.arange(0.05, 0.55, 0.05).astype(float):
            is_exist(
                f"{OUTPUT_DIR}/{data_flag}/rate_{rate:.2f}/resnet18_{NOW}_{ENV}.pth"
            )
            is_exist(
                f"{OUTPUT_DIR}/{data_flag}/rate_{rate:.2f}/indices_{NOW}_{ENV}.pkl"
            )

        NOW = "2025-03-16-03-16-23"
        ENV = "kogoro"
        for rate in np.arange(0.5, 0.75, 0.05).astype(float):
            for target_class in range(1):
                all += 2
                exist += is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/resnet18_{NOW}_{ENV}.pth"
                )
                exist += is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/indices_{NOW}_{ENV}.pkl"
                )

        NOW = "2025-03-16-03-17-32"
        for rate in np.arange(0.75, 1.05, 0.05).astype(float):
            for target_class in range(1):
                all += 2
                exist += is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/resnet18_{NOW}_{ENV}.pth"
                )
                exist += is_exist(
                    f"{OUTPUT_DIR}/{data_flag}/class_{target_class}_rate_{rate:.2f}/indices_{NOW}_{ENV}.pkl"
                )
    print(f"All: {all}, Exist: {exist}")


def main(batch_size=128, device="cuda:0"):
    for data_flag in DATAFLAGS:
        NOW = "2025-03-16-03-13-37"
        ENV = "ayame"
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
        default_model = default_model.to(device)

        for rate in np.arange(0.05, 0.1, 0.05).astype(float):
            output_dir = os.path.join(OUTPUT_DIR, data_flag, f"rate_{rate:.02f}")

            retrain_model = get_resnet18(
                num_channels=num_channels, num_classes=num_classes
            )
            retrain_model_path = os.path.join(output_dir, f"resnet18_{NOW}_{ENV}.pth")
            retrain_model.load_state_dict(torch.load(retrain_model_path))
            logger_regular.info(f"Model loaded at {retrain_model_path}")
            retrain_model = retrain_model.to(device)

            indices_path = os.path.join(output_dir, f"indices_{NOW}_{ENV}.pkl")
            with open(indices_path, "rb") as f:
                unseen_indices, train_indices = pickle.load(f)
            logger_regular.info(f"Indices loaded at {indices_path}")
            unseen_dataset = torch.utils.data.Subset(train_dataset, unseen_indices)

            default_outputs, default_targets = get_targets_and_outputs(
                default_model, unseen_dataset, batch_size, device
            )
            default_outputs = default_outputs.to(device)
            default_targets = default_targets.to(device)
            retrain_outputs, retrain_targets = get_targets_and_outputs(
                retrain_model, unseen_dataset, batch_size, device
            )
            retrain_outputs = retrain_outputs.to(device)
            retrain_targets = retrain_targets.to(device)
            default_fa = multiclass_accuracy(
                default_outputs, default_targets, num_classes=num_classes
            )
            retrain_fa = multiclass_accuracy(
                retrain_outputs, retrain_targets, num_classes=num_classes
            )

            logger_regular.info(
                f"rate: {rate} | default fa: {default_fa}, retrain fa: {retrain_fa}"
            )

        NOW = "2025-03-16-03-16-23"
        ENV = "kogoro"
        for rate in np.arange(0.5, 0.54, 0.05).astype(float):
            for target_class in range(1):
                output_dir = os.path.join(
                    OUTPUT_DIR, data_flag, f"class_{target_class}_rate_{rate:.02f}"
                )

                retrain_model = get_resnet18(
                    num_channels=num_channels, num_classes=num_classes
                )
                retrain_model_path = os.path.join(
                    output_dir, f"resnet18_{NOW}_{ENV}.pth"
                )
                retrain_model.load_state_dict(torch.load(retrain_model_path))
                logger_regular.info(f"Model loaded at {retrain_model_path}")
                retrain_model = retrain_model.to(device)

                indices_path = os.path.join(output_dir, f"indices_{NOW}_{ENV}.pkl")
                with open(indices_path, "rb") as f:
                    unseen_indices, train_indices = pickle.load(f)
                logger_regular.info(f"Indices loaded at {indices_path}")
                unseen_dataset = torch.utils.data.Subset(train_dataset, unseen_indices)

                default_outputs, default_targets = get_targets_and_outputs(
                    default_model, unseen_dataset, batch_size, device
                )
                default_outputs = default_outputs.to(device)
                default_targets = default_targets.to(device)
                retrain_outputs, retrain_targets = get_targets_and_outputs(
                    retrain_model, unseen_dataset, batch_size, device
                )
                retrain_outputs = retrain_outputs.to(device)
                retrain_targets = retrain_targets.to(device)
                default_fa = multiclass_accuracy(
                    default_outputs, default_targets, num_classes=num_classes
                )
                retrain_fa = multiclass_accuracy(
                    retrain_outputs, retrain_targets, num_classes=num_classes
                )

                logger_regular.info(
                    f"class: {target_class}, rate: {rate} | default fa: {default_fa}, retrain fa: {retrain_fa}"
                )


# main()
check()
