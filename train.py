import os
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.dataset import (
    DATA_FLAGS,
    get_dataset,
    get_num_channels,
    get_num_classes,
    split_dataset_by_class,
    split_dataset_by_rate,
)
from src.logger import logger_regular
from src.misc import fix_seeds, now
from src.model import get_model
from src.train import train

fix_seeds(0)


def generate_default_model(
    output_dir: str,
    dataset_dir: str,
    data_flag: str,
    arch: str,
    train_rate: float,
    val_rate: float,
    timestamp: str,
    num_epochs: int,
    device: str,
):
    output_dir = os.path.join(output_dir, data_flag, "default")
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_default_model")
    logger_regular.info(f"data_flag: {data_flag}, num_epochs: {num_epochs}\n")

    train_dataset, val_dataset, test_dataset = get_dataset(
        data_flag=data_flag,
        dataset_dir=dataset_dir,
        train_rate=train_rate,
        val_rate=val_rate,
    )
    num_channels = get_num_channels(data_flag=data_flag)
    num_classes = get_num_classes(data_flag=data_flag)

    model = get_model(arch=arch, num_channels=num_channels, num_classes=num_classes)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{timestamp}")
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
    model_path = os.path.join(output_dir, f"{arch}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


def generate_model_with_dataset_reduced_by_rate(
    output_dir: str,
    dataset_dir: str,
    data_flag: str,
    arch: str,
    train_rate: float,
    val_rate: float,
    timestamp: str,
    num_epochs: int,
    device: str,
    rate: float,
):
    output_dir = os.path.join(output_dir, data_flag, f"rate_{rate:.02f}")
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_model_with_dataset_reduced_by_rate")
    logger_regular.info(
        f"data_flag: {data_flag}, num_epochs: {num_epochs}, rate: {rate}\n"
    )

    train_dataset, val_dataset, test_dataset = get_dataset(
        data_flag=data_flag,
        dataset_dir=dataset_dir,
        train_rate=train_rate,
        val_rate=val_rate,
    )
    num_channels = get_num_channels(data_flag=data_flag)
    num_classes = get_num_classes(data_flag=data_flag)

    indices_path = os.path.join(output_dir, f"indices_{timestamp}.pkl")
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

    model = get_model(arch=arch, num_channels=num_channels, num_classes=num_classes)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{timestamp}")
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
    model_path = os.path.join(output_dir, f"{arch}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


def generate_model_with_dataset_excluded_by_class(
    output_dir: str,
    dataset_dir: str,
    data_flag: str,
    arch: str,
    train_rate: float,
    val_rate: float,
    timestamp: str,
    num_epochs: int,
    device: str,
    target_class: int,
    rate: float,
):
    output_dir = os.path.join(
        output_dir, data_flag, f"class_{target_class}_rate_{rate:.2f}"
    )
    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("generate_model_with_dataset_excluded_by_class")
    logger_regular.info(
        f"data_flag: {data_flag}, num_epochs: {num_epochs}, class: {target_class}, rate: {rate}\n"
    )

    train_dataset, val_dataset, test_dataset = get_dataset(
        data_flag=data_flag,
        dataset_dir=dataset_dir,
        train_rate=train_rate,
        val_rate=val_rate,
    )
    num_channels = get_num_channels(data_flag=data_flag)
    num_classes = get_num_classes(data_flag=data_flag)

    indices_path = os.path.join(output_dir, f"indices_{timestamp}.pkl")
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

    model = get_model(arch=arch, num_channels=num_channels, num_classes=num_classes)

    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, f"Tensorboard_Results_{timestamp}")
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
    model_path = os.path.join(output_dir, f"{arch}_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger_regular.info(f"Model saved at {model_path}")

    logger_regular.info(
        "=============================================================="
    )
    logger_regular.info("")


def main(
    output_dir: str,
    dataset_dir: str,
    data_flags: list[str],
    archs: list[str],
    train_rate: float,
    val_rate: float,
    reducing_rates: np.ndarray,
    class_rates: np.ndarray,
    device: str,
    timestamp: str,
):
    start_time = time.perf_counter()
    logger_regular.info("main")

    for data_flag in data_flags:
        for arch in archs:
            generate_default_model(
                output_dir=output_dir,
                dataset_dir=dataset_dir,
                data_flag=data_flag,
                arch=arch,
                train_rate=train_rate,
                val_rate=val_rate,
                timestamp=timestamp,
                num_epochs=100,
                device=device,
            )

            for rate in reducing_rates:
                generate_model_with_dataset_reduced_by_rate(
                    output_dir=output_dir,
                    dataset_dir=dataset_dir,
                    data_flag=data_flag,
                    arch=arch,
                    train_rate=train_rate,
                    val_rate=val_rate,
                    timestamp=timestamp,
                    num_epochs=100,
                    rate=rate,
                    device=device,
                )

            for rate in class_rates:
                for target_class in range(get_num_classes(data_flag=data_flag)):
                    generate_model_with_dataset_excluded_by_class(
                        output_dir=output_dir,
                        dataset_dir=dataset_dir,
                        data_flag=data_flag,
                        arch=arch,
                        train_rate=train_rate,
                        val_rate=val_rate,
                        timestamp=timestamp,
                        num_epochs=100,
                        target_class=target_class,
                        rate=rate,
                        device=device,
                    )

    logger_regular.info(f"{time.perf_counter() - start_time}s.")


main(
    output_dir="/nas/keito/ML_diff_experiment/output",
    dataset_dir="dataset",
    data_flags=DATA_FLAGS,
    archs=["resnet18"],
    train_rate=0.4,
    val_rate=0.2,
    reducing_rates=np.arange(0.05, 1.01, 0.05).astype(float),
    class_rates=np.arange(0.05, 1.01, 0.05).astype(float),
    device="cuda:0",
    timestamp=now(),
)
