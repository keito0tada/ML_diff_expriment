import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torcheval.metrics.functional import multiclass_accuracy, multiclass_auroc
from tqdm import trange

from src.logger import logger_regular


def train(
    model: nn.Module,
    num_classes: int,
    train_dataset,
    val_dataset,
    test_dataset,
    writer: SummaryWriter,
    device: str,
    num_epochs=100,
    batch_size=64,
):
    logger_regular.info("=====> train")
    start_time = time.perf_counter()

    lr = 0.001
    gamma = 0.1
    milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) % batch_size == 1,
    )
    train_loader_for_eval = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )
    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    train_target = torch.tensor([y for _, y in train_dataset], dtype=torch.int64)
    val_target = torch.tensor([y for _, y in val_dataset], dtype=torch.int64)
    test_target = torch.tensor([y for _, y in test_dataset], dtype=torch.int64)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    # writer = SummaryWriter(log_dir=os.path.join(output_dir, "Tensorboard_Results"))

    # best_acc = 0
    # best_acc_epoch = 0
    best_auc = 0
    best_auc_epoch = 0
    # best_loss = 1e8
    # best_loss_epoch = 0
    best_model = deepcopy(model)

    for epoch in trange(num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        train_loss, train_outputs = test(
            model, train_loader_for_eval, criterion, device
        )
        train_acc = multiclass_accuracy(
            torch.cat(train_outputs, dim=0).to(device),
            train_target.to(device),
            num_classes=num_classes,
        ).item()
        train_auc = multiclass_auroc(
            torch.cat(train_outputs, dim=0).to(device),
            train_target.to(device),
            num_classes=num_classes,
            average="macro",
        ).item()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_auc", train_auc, epoch)
        logger_regular.info(
            f"Epoch {epoch}: train_loss: {train_loss}, train_acc: {train_acc}, train_auc: {train_auc}"
        )

        val_loss, val_outputs = test(model, val_loader, criterion, device)
        val_acc = multiclass_accuracy(
            torch.cat(val_outputs, dim=0).to(device),
            val_target.to(device),
            num_classes=num_classes,
        ).item()
        val_auc = multiclass_auroc(
            torch.cat(val_outputs, dim=0).to(device),
            val_target.to(device),
            num_classes=num_classes,
            average="macro",
        ).item()
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_auc", val_auc, epoch)
        logger_regular.info(
            f"Epoch {epoch}: val_loss: {val_loss}, val_acc: {val_acc}, val_auc: {val_auc}"
        )

        # if val_acc > best_acc:
        #     best_acc_epoch = epoch
        #     best_acc = val_acc
        #     best_model = deepcopy(model)
        #     logger_regular.info(f"cur_best_acc: { best_acc}")
        #     logger_regular.info(f"cur_best_epoch: {best_acc_epoch}")
        # if val_loss < best_loss:
        #     best_epoch = epoch
        #     best_loss = val_loss
        #     best_model = deepcopy(model)
        #     logger_regular.info(f"cur_best_loss: {best_loss}")
        #     logger_regular.info(
        #         f"cur_best_epoch: {
        #         best_epoch}"
        #     )
        if val_auc > best_auc:
            best_auc_epoch = epoch
            best_auc = val_auc
            best_model = deepcopy(model)
            logger_regular.info(f"cur_best_auc: { best_auc}")
            logger_regular.info(f"cur_best_epoch: {best_auc_epoch}")

    test_loss, test_outputs = test(best_model, test_loader, criterion, device)

    test_acc = multiclass_accuracy(
        torch.cat(test_outputs, dim=0).to(device),
        test_target.to(device),
        num_classes=num_classes,
    ).item()
    test_auc = multiclass_auroc(
        torch.cat(test_outputs, dim=0).to(device),
        test_target.to(device),
        num_classes=num_classes,
        average="macro",
    ).item()
    logger_regular.info(
        f"test_loss: {test_loss}, test_acc: {test_acc}, test_auc: {test_auc}"
    )

    writer.close()

    logger_regular.info(f"==> Finished in {time.perf_counter() - start_time:.2f} s.")

    return best_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    losses = []

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return sum(losses)


def test(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion, device: str
):
    losses = []
    all_outputs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            outputs = nn.Softmax(dim=1)(outputs).to(device)

            losses.append(loss.item())
            all_outputs.append(outputs)

    return sum(losses), all_outputs
