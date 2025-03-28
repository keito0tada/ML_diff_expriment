import medmnist
import numpy as np
import torch
import torchvision
from medmnist import INFO

from src.logger import logger_regular

DATA_FLAGS = [
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


def get_num_channels(data_flag: str):
    if data_flag == "mnist":
        return 1
    elif data_flag == "cifar10":
        return 3
    elif data_flag == "cifar100":
        return 3
    else:
        return medmnist.INFO[data_flag]["n_channels"]


def get_num_classes(data_flag: str):
    if data_flag == "mnist":
        return 10
    elif data_flag == "cifar10":
        return 10
    elif data_flag == "cifar100":
        return 100
    else:
        return len(medmnist.INFO[data_flag]["label"])


def get_medmnist_dataset(data_flag: str, size: int):
    info = INFO[data_flag]
    task = info["task"]
    num_channels = info["n_channels"]
    num_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    data_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = DataClass(
        split="train", transform=data_transform, download=True, size=size
    )
    val_dataset = DataClass(
        split="val", transform=data_transform, download=True, size=size
    )
    test_dataset = DataClass(
        split="test", transform=data_transform, download=True, size=size
    )

    # print(train_dataset.imgs.shape)
    # print(train_dataset.imgs.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.imgs.std(axis=(0, 1, 2)) / 255)

    # print(np.array([x for x, y in train_dataset]).shape)
    # print(np.array([x for x, y in train_dataset]).mean(axis=(0, 2, 3)))
    # print(np.array([x for x, y in train_dataset]).std(axis=(0, 2, 3)))

    logger_regular.info(
        f"{data_flag} | train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    logger_regular.info(
        f"{data_flag} | task: {task}, num_channels: {num_channels}, num_classes: {num_classes}, size: {size}"
    )
    return train_dataset, val_dataset, test_dataset, task, num_channels, num_classes


def get_medmnist_dataset_with_single_label(data_flag: str, size: int):
    train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
        get_medmnist_dataset(data_flag, size)
    )
    train_data_x = []
    train_data_y = []
    for X, y in train_dataset:
        train_data_x.append(X)
        train_data_y.append(y[0])

    val_data_x = []
    val_data_y = []
    for X, y in val_dataset:
        val_data_x.append(X)
        val_data_y.append(y[0])

    test_data_x = []
    test_data_y = []
    for X, y in test_dataset:
        test_data_x.append(X)
        test_data_y.append(y[0])

    train_dataset = torch.utils.data.TensorDataset(
        torch.stack(train_data_x), torch.tensor(train_data_y, dtype=torch.int64)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.stack(val_data_x), torch.tensor(val_data_y, dtype=torch.int64)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.stack(test_data_x), torch.tensor(test_data_y, dtype=torch.int64)
    )

    return train_dataset, val_dataset, test_dataset, task, num_channels, num_classes


def get_mnist_dataset(dataset_dir: str):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        download=True,
        transform=transforms,
    )

    logger_regular.info(
        f"MNIST | train: {len(train_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, test_dataset


def get_cifar10_dataset(dataset_dir: str):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.49139968, 0.48215841, 0.44653091),
                (0.24703223, 0.24348513, 0.26158784),
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        download=True,
        transform=transforms,
    )

    # print(train_dataset.data.shape)
    # print(train_dataset.data.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.data.std(axis=(0, 1, 2)) / 255)

    # print(np.array([x for x, y in train_dataset]).shape)
    # print(np.array([x for x, y in train_dataset]).mean(axis=(0, 2, 3)))
    # print(np.array([x for x, y in train_dataset]).std(axis=(0, 2, 3)))

    logger_regular.info(
        f"CIFAR10 | train: {len(train_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, test_dataset


def get_cifar100_dataset(dataset_dir: str):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.50707516, 0.48654887, 0.44091784),
                (0.26733429, 0.25643846, 0.27615047),
            ),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=False, download=True, transform=transforms
    )

    # print(train_dataset.data.shape)
    # print(train_dataset.data.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.data.std(axis=(0, 1, 2)) / 255)

    # print(np.array([x for x, y in train_dataset]).shape)
    # print(np.array([x for x, y in train_dataset]).mean(axis=(0, 2, 3)))
    # print(np.array([x for x, y in train_dataset]).std(axis=(0, 2, 3)))

    logger_regular.info(
        f"CIFAR100 | train: {len(train_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, test_dataset


def split_dataset_by_rate(dataset, rate=0.1):
    indices = np.arange(len(dataset)).tolist()
    np.random.shuffle(indices)
    logger_regular.info(indices[:5])
    # unseen_dataset = torch.utils.data.Subset(dataset, indices[: int(len(dataset) * rate)])
    # train_dataset = torch.utils.data.Subset(dataset, indices[int(len(dataset) * rate) :])
    # print(f"Splitted | unseen_dataset: {len(unseen_dataset)}, train_dataset: {len(train_dataset)}")
    return (
        indices[: int(len(dataset) * rate)],
        indices[int(len(dataset) * rate) :],
    )


def split_dataset_by_class(dataset, target_classes: list, rate: float = 0):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][1] in target_classes:
            indices.append(i)
    np.random.shuffle(indices)
    logger_regular.info(indices[:5])
    # dataset1 = torch.utils.data.Subset(dataset, indices[: int(len(indices) * rate)])
    # dataset2 = torch.utils.data.Subset(
    #     dataset,
    #     list(set(range(len(dataset))) - set(indices[: int(len(indices) * rate)])),
    # )
    # print(f"Splitted | dataset1: {len(dataset1)}, dataset2: {len(dataset2)}")
    return indices[: int(len(indices) * rate)], list(
        set(range(len(dataset))) - set(indices[: int(len(indices) * rate)])
    )


def split_dataset_to_train_val_test(
    datasets: list[torch.utils.data.Dataset], train_rate: float, val_rate: float
):
    concatted_dataset = torch.utils.data.ConcatDataset(datasets)
    indices = np.arange(len(concatted_dataset)).tolist()
    last_train_index = int(len(concatted_dataset) * train_rate)
    last_val_index = int(len(concatted_dataset) * (train_rate + val_rate))
    return (
        torch.utils.data.Subset(concatted_dataset, indices[:last_train_index]),
        torch.utils.data.Subset(
            concatted_dataset, indices[last_train_index:last_val_index]
        ),
        torch.utils.data.Subset(concatted_dataset, indices[last_val_index:]),
    )


def get_dataset(data_flag: str, dataset_dir: str, train_rate: float, val_rate: float):
    if data_flag == "mnist":
        train_dataset, test_dataset = get_mnist_dataset(dataset_dir=dataset_dir)
        train_dataset, val_dataset, test_dataset = split_dataset_to_train_val_test(
            datasets=[train_dataset, test_dataset],
            train_rate=train_rate,
            val_rate=val_rate,
        )
    elif data_flag == "cifar10":
        train_dataset, test_dataset = get_cifar10_dataset(dataset_dir=dataset_dir)
        train_dataset, val_dataset, test_dataset = split_dataset_to_train_val_test(
            datasets=[train_dataset, test_dataset],
            train_rate=train_rate,
            val_rate=val_rate,
        )
    elif data_flag == "cifar100":
        train_dataset, test_dataset = get_cifar100_dataset(dataset_dir=dataset_dir)
        train_dataset, val_dataset, test_dataset = split_dataset_to_train_val_test(
            datasets=[train_dataset, test_dataset],
            train_rate=train_rate,
            val_rate=val_rate,
        )
    else:
        train_dataset, val_dataset, test_dataset, task, num_channels, num_classes = (
            get_medmnist_dataset_with_single_label(data_flag, 224)
        )
        train_dataset, val_dataset, test_dataset = split_dataset_to_train_val_test(
            datasets=[train_dataset, val_dataset, test_dataset],
            train_rate=train_rate,
            val_rate=val_rate,
        )
    return train_dataset, val_dataset, test_dataset
