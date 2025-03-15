import torch
from torcheval.metrics.functional import multiclass_accuracy

from train import test


def get_acc(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    batch_size: int,
    device: str,
):
    targets = torch.tensor([y for _, y in dataset], dtype=torch.int64)
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    loss, outputs = test(model, data_loader, criterion, device)
    return multiclass_accuracy(
        torch.cat(outputs, dim=0).to(device),
        targets.to(device),
        num_classes=num_classes,
    ).item()
