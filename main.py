import os

from dataset import (
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_medmnist_dataset_with_single_label,
    get_mnist_dataset,
    split_dataset_by_class,
    split_dataset_by_rate,
)
from misc import fix_seeds, now
from model import get_resnet18
from train import train

# OUTPUT_DIR = "output"
OUTPUT_DIR = "/nas/keito/ML_diff_experiment/output"
ENV = "ayame"

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

fix_seeds(0)


def main(data_flag: str, num_epochs: int = 100, device="cuda:0"):
    NOW = now()
    print("===============================")
    print(f"{data_flag}, {num_epochs}, default\n")

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
    print(train_dataset[0][0].shape)

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)
    train(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}"),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    with open(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}", "experiment.txt"), "w"
    ) as f:
        f.write(f"{data_flag}, {num_epochs}, default\n")

    print("==> Done.")


def main_rate(data_flag: str, num_epochs: int = 100, device="cuda:0", rate=0.1):
    NOW = now()
    print("===============================")
    print("{data_flag}, {num_epochs}, rate, {rate}\n")

    if data_flag == "mnist":
        num_channels = 1
        num_classes = 10
        train_dataset, val_dataset, test_dataset = get_mnist_dataset()
        unseen_dataset, train_dataset = split_dataset_by_rate(train_dataset, rate)
    elif data_flag == "cifar10":
        num_channels = 3
        num_classes = 10
        train_dataset, val_dataset, test_dataset = get_cifar10_dataset()
        unseen_dataset, train_dataset = split_dataset_by_rate(train_dataset, rate)
    elif data_flag == "cifar100":
        num_channels = 3
        num_classes = 100
        train_dataset, val_dataset, test_dataset = get_cifar100_dataset()
        unseen_dataset, train_dataset = split_dataset_by_rate(train_dataset, rate)
    else:
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag)
        )
        unseen_dataset, train_dataset = split_dataset_by_rate(train_dataset, rate)

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    train(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}"),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    with open(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}", "experiment.txt"), "w"
    ) as f:
        f.write(f"{data_flag}, {num_epochs}, rate, {rate}\n")

    print("==> Done.")


def main_class(
    data_flag: str, num_epochs: int = 100, device="cuda:0", target_classes=[0], rate=1.0
):
    NOW = now()
    print("===============================")
    print(f"{data_flag}, {num_epochs}, classes, {target_classes}, rate, {rate}\n")

    if data_flag == "mnist":
        num_channels = 1
        num_classes = 10
        train_dataset, val_dataset, test_dataset = get_mnist_dataset()
        unseen_dataset, train_dataset = split_dataset_by_class(
            train_dataset, target_classes, rate
        )
    elif data_flag == "cifar10":
        num_channels = 3
        num_classes = 10
        train_dataset, val_dataset, test_dataset = get_cifar10_dataset()
        unseen_dataset, train_dataset = split_dataset_by_class(
            train_dataset, target_classes, rate
        )
    elif data_flag == "cifar100":
        num_channels = 3
        num_classes = 100
        train_dataset, val_dataset, test_dataset = get_cifar100_dataset()
        unseen_dataset, train_dataset = split_dataset_by_class(
            train_dataset, target_classes, rate
        )
    else:
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag)
        )
        unseen_dataset, train_dataset = split_dataset_by_class(
            train_dataset, target_classes, rate
        )

    model = get_resnet18(num_channels=num_channels, num_classes=num_classes)

    train(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}"),
        model,
        num_classes,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        num_epochs=num_epochs,
    )

    with open(
        os.path.join(OUTPUT_DIR, data_flag, f"{NOW}_{ENV}", "experiment.txt"), "w"
    ) as f:
        f.write(f"{data_flag}, {num_epochs}, classes, {target_classes}, rate, {rate}\n")

    print("==> Done.")


# for data_flag in DATASETS:
#     main(data_flag, 100)

#     for rate in np.arange(0.05, 0.1, 0.05).astype(float):
#         main_rate(data_flag, 100, rate=rate)

#     for rate in [0.5, 1.0]:
#         for target_classes in [[0]]:
#             main_class(data_flag, 100, target_classes=target_classes, rate=rate)

main("mnist", 10)
