import os

import numpy as np

from dataset import (
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_medmnist_dataset_with_single_label,
    get_mnist_dataset,
    split_dataset_by_class,
    split_dataset_by_rate,
)
from logger import logger_regular
from misc import fix_seeds, now
from model import get_resnet18
from train import train

fix_seeds(0)

ENV = "ayame"

# OUTPUT_DIR = "output"
OUTPUT_DIR = "/nas/keito/ML_diff_experiment/output"

DATASETS = [
    "mnist",
    # "cifar10",
    # "cifar100",
    # "pathmnist",
    # "dermamnist",
    # "octmnist",
    # "pneumoniamnist",
    # "retinamnist",
    # "breastmnist",
    # "bloodmnist",
    # "tissuemnist",
    # "organamnist",
    # "organcmnist",
    # "organsmnist",
]

NOW = now()


def generate_default_model(data_flag: str, num_epochs: int = 100, device="cuda:0"):
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_default_model")
    logger_regular.info(f"data_flag: {data_flag}, num_epochs: {num_epochs}\n")

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
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag)
        )

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)
    train(
        os.path.join(OUTPUT_DIR, data_flag, f"default_{NOW}_{ENV}"),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    logger_regular.info(
        "=============================================================="
    )


def generate_model_with_dataset_reduced_by_rate(
    data_flag: str, num_epochs: int = 100, device="cuda:0", rate=0.1
):
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_model_with_dataset_reduced_by_rate")
    logger_regular.info(
        f"data_flag: {data_flag}, num_epochs: {num_epochs}, rate: {rate}\n"
    )

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
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag)
        )

    unseen_dataset, train_dataset = split_dataset_by_rate(train_dataset, rate)

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    train(
        os.path.join(OUTPUT_DIR, data_flag, f"rate_{rate}_{NOW}_{ENV}"),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    logger_regular.info(
        "=============================================================="
    )


def generate_model_with_dataset_excluded_by_class(
    data_flag: str, num_epochs: int = 100, device="cuda:0", target_class=0, rate=1.0
):
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_model_with_dataset_excluded_by_class")
    logger_regular.info(
        f"data_flag: {data_flag}, num_epochs: {num_epochs}, class: {target_class}, rate: {rate}\n"
    )

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
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag)
        )

    if target_class >= num_classes:
        logger_regular.warning("Invalid class.")
        return

    unseen_dataset, train_dataset = split_dataset_by_class(
        train_dataset, [target_class], rate
    )

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    train(
        os.path.join(
            OUTPUT_DIR, data_flag, f"class_{target_class}_rate_{rate}_{NOW}_{ENV}"
        ),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


# def main_invalid_class(
#     data_flag: str,
#     num_epochs: int = 100,
#     device="cuda:0",
#     target_class=0,
#     invalid_class=1,
#     rate=1.0,
# ):
#     NOW = now()
#     print("===============================")
#     print(
#         f"{data_flag} | epoch: {num_epochs}, classes: {target_class}, invalid: {invalid_class}, rate: {rate}\n"
#     )

#     if data_flag == "mnist":
#         num_channels = 1
#         num_classes = 10
#         train_dataset, val_dataset, test_dataset = get_mnist_dataset()
#         unseen_dataset, train_dataset = split_dataset_by_class(
#             train_dataset, [target_class], rate
#         )
#     elif data_flag == "cifar10":
#         num_channels = 3
#         num_classes = 10
#         train_dataset, val_dataset, test_dataset = get_cifar10_dataset()
#         unseen_dataset, train_dataset = split_dataset_by_class(
#             train_dataset, [target_class], rate
#         )
#     elif data_flag == "cifar100":
#         num_channels = 3
#         num_classes = 100
#         train_dataset, val_dataset, test_dataset = get_cifar100_dataset()
#         unseen_dataset, train_dataset = split_dataset_by_class(
#             train_dataset, [target_class], rate
#         )
#     else:
#         train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
#             get_medmnist_dataset_with_single_label(data_flag)
#         )
#         unseen_dataset, train_dataset = split_dataset_by_class(
#             train_dataset, [target_class], rate
#         )

#     model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

#     train(
#         os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}"),
#         model,
#         num_classes,
#         train_dataset,
#         val_dataset,
#         test_dataset,
#         device,
#         num_epochs=num_epochs,
#     )

#     with open(
#         os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}", "experiment.txt"), "w"
#     ) as f:
#         f.write(f"{data_flag}, {num_epochs}, classes, {target_class}, rate, {rate}\n")

#     print("==> Done.")


for data_flag in DATASETS:
    generate_default_model(data_flag, 1)

    for rate in np.arange(0.05, 0.5, 0.05).astype(float):
        generate_model_with_dataset_reduced_by_rate(data_flag, 1, rate=rate)

    for rate in [0.5, 1.0]:
        for target_class in range(13):
            generate_model_with_dataset_excluded_by_class(
                data_flag, 1, target_class=target_class, rate=rate
            )
