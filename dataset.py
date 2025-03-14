import medmnist
import numpy as np
import torch
import torchvision
from medmnist import INFO
from sklearn.model_selection import train_test_split

DATASET_DIR = "data"
RANDOM_STATE = 0


def get_medmnist_dataset(data_flag="pathmnist", size=28):
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

    print(
        f"{data_flag} | train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    print(
        f"{data_flag} | task: {task}, num_channels: {num_channels}, num_classes: {num_classes}"
    )
    return train_dataset, val_dataset, test_dataset, task, num_channels, num_classes


def get_medmnist_dataset_with_single_label(data_flag="pathmnist", size=28):
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


def get_mnist_dataset(val_rate=0.2):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATASET_DIR,
        train=False,
        download=True,
        transform=transforms,
    )

    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=val_rate,
        stratify=train_dataset.targets,
        random_state=RANDOM_STATE,
    )

    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    print(
        f"MNIST | train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def get_cifar10_dataset(val_rate=0.2):
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
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_DIR,
        train=False,
        download=True,
        transform=transforms,
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.stack([x for x, y in train_dataset]),
        torch.tensor([y for x, y in train_dataset], dtype=torch.int64),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.stack([x for x, y in test_dataset]),
        torch.tensor([y for x, y in test_dataset], dtype=torch.int64),
    )

    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=val_rate,
        stratify=[y for x, y in train_dataset],
        random_state=RANDOM_STATE,
    )

    # print(train_dataset.data.shape)
    # print(train_dataset.data.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.data.std(axis=(0, 1, 2)) / 255)

    # print(np.array([x for x, y in train_dataset]).shape)
    # print(np.array([x for x, y in train_dataset]).mean(axis=(0, 2, 3)))
    # print(np.array([x for x, y in train_dataset]).std(axis=(0, 2, 3)))

    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    print(
        f"CIFAR10 | train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def get_cifar100_dataset(val_rate=0.2):
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
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transforms,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=DATASET_DIR, train=False, download=True, transform=transforms
    )

    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=val_rate,
        stratify=train_dataset.targets,
        random_state=RANDOM_STATE,
    )

    # print(train_dataset.data.shape)
    # print(train_dataset.data.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.data.std(axis=(0, 1, 2)) / 255)

    # print(np.array([x for x, y in train_dataset]).shape)
    # print(np.array([x for x, y in train_dataset]).mean(axis=(0, 2, 3)))
    # print(np.array([x for x, y in train_dataset]).std(axis=(0, 2, 3)))

    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    print(
        f"CIFAR100 | train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


def split_dataset_by_rate(dataset: torch.utils.data.Dataset, rate=0.1):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset1 = torch.utils.data.Subset(dataset, indices[: int(len(dataset) * rate)])
    dataset2 = torch.utils.data.Subset(dataset, indices[int(len(dataset) * rate) :])
    print(f"Splitted | dataset1: {len(dataset1)}, dataset2: {len(dataset2)}")
    return dataset1, dataset2


def split_dataset_by_class(
    dataset: torch.utils.data.Dataset, target_classes: list, rate: float = 0
):
    indices = []
    for i in range(len(dataset)):
        if dataset[i][1] in target_classes:
            indices.append(i)
    np.random.shuffle(indices)
    print(indices[:5])
    dataset1 = torch.utils.data.Subset(dataset, indices[: int(len(indices) * rate)])
    dataset2 = torch.utils.data.Subset(
        dataset,
        list(set(range(len(dataset))) - set(indices[: int(len(indices) * rate)])),
    )
    print(f"Splitted | dataset1: {len(dataset1)}, dataset2: {len(dataset2)}")
    return dataset1, dataset2
