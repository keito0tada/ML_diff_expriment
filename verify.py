import os
import pickle

import numpy as np
import torch
from torcheval.metrics.functional import multiclass_accuracy

from src.dataset import (
    DATA_FLAGS,
    get_dataset,
    get_num_channels,
    get_num_classes,
)
from src.logger import logger_regular
from src.model import get_model, get_resnet18
from src.verify import get_targets_and_outputs


def is_exist(path: str):
    if os.path.isfile(path):
        print(f"o | {path}")
        return True
    else:
        print(f"x | {path}")
        return False


def check_files(
    output_dir: str,
    data_flags: list[str],
    archs: list[str],
    reduced_rates: np.ndarray,
    class_rates: np.ndarray,
    timestamp: str,
):
    all = 0
    exist = 0
    for data_flag in data_flags:
        print(f"==================== {data_flag} ====================")

        for arch in archs:
            print(f"     ==================== {arch} ====================")

            all += 1
            exist += is_exist(
                f"{output_dir}/{data_flag}/default/{archs}_{timestamp}.pth"
            )

            for rate in reduced_rates:
                all += 2
                exist += is_exist(
                    f"{output_dir}/{data_flag}/rate_{rate:.2f}/{arch}_{timestamp}.pth"
                )
                exist += is_exist(
                    f"{output_dir}/{data_flag}/rate_{rate:.2f}/indices_{timestamp}.pkl"
                )

            for rate in class_rates:
                for target_class in range(get_num_classes(data_flag=data_flag)):
                    all += 2
                    exist += is_exist(
                        f"{output_dir}/{data_flag}/class_{target_class}_rate_{rate:.2f}/{arch}_{timestamp}.pth"
                    )
                    exist += is_exist(
                        f"{output_dir}/{data_flag}/class_{target_class}_rate_{rate:.2f}/indices_{timestamp}.pkl"
                    )

    print(f"All: {all}, Exist: {exist}")


def main(
    output_dir: str,
    dataset_dir: str,
    data_flags: list[str],
    archs: list[str],
    train_rate: float,
    val_rate: float,
    reduced_rates: np.ndarray,
    class_rates: np.ndarray,
    timestamp: str,
    batch_size: int,
    device: str,
):
    for data_flag in data_flags:
        for arch in archs:
            train_dataset, val_dataset, test_dataset = get_dataset(
                data_flag=data_flag,
                dataset_dir=dataset_dir,
                train_rate=train_rate,
                val_rate=val_rate,
            )
            num_channels = get_num_channels(data_flag=data_flag)
            num_classes = get_num_classes(data_flag=data_flag)

            default_model = get_model(
                arch=arch, num_channels=num_channels, num_classes=num_classes
            )
            default_model.load_state_dict(
                torch.load(f"{output_dir}/{data_flag}/default/{arch}_{timestamp}.pth")
            )
            default_model = default_model.to(device)

            for rate in reduced_rates:
                output_dir = os.path.join(output_dir, data_flag, f"rate_{rate:.02f}")

                retrain_model = get_resnet18(
                    num_channels=num_channels, num_classes=num_classes
                )
                retrain_model_path = os.path.join(output_dir, f"{arch}_{timestamp}.pth")
                retrain_model.load_state_dict(torch.load(retrain_model_path))
                logger_regular.info(f"Model loaded at {retrain_model_path}")
                retrain_model = retrain_model.to(device)

                indices_path = os.path.join(output_dir, f"indices_{timestamp}.pkl")
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

            for rate in class_rates:
                for target_class in range(1):
                    output_dir = os.path.join(
                        output_dir, data_flag, f"class_{target_class}_rate_{rate:.02f}"
                    )

                    retrain_model = get_resnet18(
                        num_channels=num_channels, num_classes=num_classes
                    )
                    retrain_model_path = os.path.join(
                        output_dir, f"{arch}_{timestamp}.pth"
                    )
                    retrain_model.load_state_dict(torch.load(retrain_model_path))
                    logger_regular.info(f"Model loaded at {retrain_model_path}")
                    retrain_model = retrain_model.to(device)

                    indices_path = os.path.join(output_dir, f"indices_{timestamp}.pkl")
                    with open(indices_path, "rb") as f:
                        unseen_indices, train_indices = pickle.load(f)
                    logger_regular.info(f"Indices loaded at {indices_path}")
                    unseen_dataset = torch.utils.data.Subset(
                        train_dataset, unseen_indices
                    )

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
