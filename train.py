import os
import pickle
import time

import medmnist
import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.dataset import (
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_medmnist_dataset_with_single_label,
    get_mnist_dataset,
    split_dataset_by_class,
    split_dataset_by_rate,
)
from src.logger import logger_regular
from src.misc import fix_seeds, now
from src.model import get_resnet18
from src.train import train

fix_seeds(0)

ENV = "ayame"
DEVICE = "cuda:0"

# OUTPUT_DIR = "output"
OUTPUT_DIR = "/nas/keito/ML_diff_experiment/output4"

DATASETS = [
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

NOW = now()
# NOW = "2025-03-15-02-13-54"


def get_num_classes(data_flag: str):
    if data_flag == "mnist":
        return 10
    elif data_flag == "cifar10":
        return 10
    elif data_flag == "cifar100":
        return 100
    else:
        return len(medmnist.INFO[data_flag]["label"])


def generate_default_model(data_flag: str, num_epochs: int = 100, device="cuda:0"):
    output_dir = os.path.join(OUTPUT_DIR, data_flag, "default")
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

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{NOW}_{ENV}")
    )

    model = train(
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        writer,
        device,
        num_epochs=num_epochs,
    )
    model_path = os.path.join(output_dir, f"resnet18_{NOW}_{ENV}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


def generate_model_with_dataset_reduced_by_rate(
    data_flag: str, num_epochs: int = 100, device="cuda:0", rate=0.1
):
    output_dir = os.path.join(OUTPUT_DIR, data_flag, f"rate_{rate:.02f}")
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

    indices_path = os.path.join(output_dir, f"indices_{NOW}_{ENV}.pkl")
    if os.path.isfile(indices_path):
        with open(indices_path, "rb") as f:
            unseen_indices, train_indices = pickle.load(f)
    else:
        unseen_indices, train_indices = split_dataset_by_rate(train_dataset, rate)
        os.makedirs(output_dir, exist_ok=True)
        with open(
            indices_path,
            "wb",
        ) as f:
            pickle.dump((unseen_indices, train_indices), f)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{NOW}_{ENV}")
    )

    model = train(
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        writer,
        device,
        num_epochs=num_epochs,
    )
    model_path = os.path.join(output_dir, f"resnet18_{NOW}_{ENV}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


def generate_model_with_dataset_excluded_by_class(
    data_flag: str, num_epochs: int = 100, device="cuda:0", target_class=0, rate=1.0
):
    output_dir = os.path.join(
        OUTPUT_DIR, data_flag, f"class_{target_class}_rate_{rate:.2f}"
    )
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

    indices_path = os.path.join(output_dir, f"indices_{NOW}_{ENV}.pkl")
    if os.path.isfile(indices_path):
        with open(indices_path, "rb") as f:
            unseen_indices, train_indices = pickle.load(f)
    else:
        unseen_indices, train_indices = split_dataset_by_class(
            train_dataset, [target_class], rate
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(indices_path, "wb") as f:
            pickle.dump((unseen_indices, train_indices), f)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{NOW}_{ENV}")
    )

    model = train(
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        writer,
        device,
        num_epochs=num_epochs,
    )
    model_path = os.path.join(output_dir, f"resnet18_{NOW}_{ENV}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

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


def main():
    start_time = time.perf_counter()
    logger_regular.info("main")

    for data_flag in DATASETS:
        generate_default_model(data_flag, num_epochs=100, device=DEVICE)

        for rate in np.arange(0.05, 0.55, 0.05).astype(float):
            generate_model_with_dataset_reduced_by_rate(
                data_flag, num_epochs=100, rate=rate, device=DEVICE
            )

        for rate in np.arange(0.5, 1.05, 0.05).astype(float):
            for target_class in range(get_num_classes(data_flag=data_flag)):
                generate_model_with_dataset_excluded_by_class(
                    data_flag,
                    num_epochs=100,
                    target_class=target_class,
                    rate=rate,
                    device=DEVICE,
                )

    logger_regular.info(f"{time.perf_counter() - start_time}s.")


def debug():
    start_time = time.perf_counter()
    logger_regular.info("debug")

    for data_flag in DATASETS:
        generate_default_model(data_flag, num_epochs=1, device=DEVICE)

        for rate in np.arange(0.05, 0.1, 0.05).astype(float):
            generate_model_with_dataset_reduced_by_rate(
                data_flag, num_epochs=1, rate=rate, device=DEVICE
            )

        for rate in np.arange(0.5, 0.55, 0.05).astype(float):
            for target_class in range(1):
                generate_model_with_dataset_excluded_by_class(
                    data_flag,
                    num_epochs=1,
                    target_class=target_class,
                    rate=rate,
                    device=DEVICE,
                )

    logger_regular.info(f"{time.perf_counter() - start_time}s.")


main()
# debug()
