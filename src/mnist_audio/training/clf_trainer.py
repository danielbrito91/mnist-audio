import json
from pathlib import Path
from time import time
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mnist_audio.utils.torch_utils import get_device


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_train_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * correct / total:.2f}%',
        })

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = 100.0 * correct / total

    return {'loss': epoch_train_loss, 'acc': epoch_train_acc}


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%',
            })

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = 100.0 * correct / total

    return {'loss': epoch_val_loss, 'acc': epoch_val_acc}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: Path,
):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    device: Optional[torch.device],
    num_epochs: int,
    lr: float,
    save_dir: Optional[Path],
):
    if device is None:
        device = get_device()

    if save_dir is None:
        save_dir = Path('checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {device}')

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    criterion = criterion.to(device)

    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    best_val_acc = 0.0
    best_model_path = save_dir / 'best_model.pth'

    print('Starting training...')
    for epoch in range(num_epochs):
        start_time = time()

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        train_history['loss'].append(train_metrics['loss'])
        train_history['acc'].append(train_metrics['acc'])
        val_history['loss'].append(val_metrics['loss'])
        val_history['acc'].append(val_metrics['acc'])

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['loss'],
                val_metrics['acc'],
                best_model_path,
            )
            print(f'New best validation accuracy: {best_val_acc:.2f}%')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['loss'],
                val_metrics['acc'],
                checkpoint_path,
            )

        epoch_time = time() - start_time

        print(
            f'Epoch {epoch + 1:3d}/{num_epochs}\n'
            f'Time: {epoch_time:.2f}s\n'
            f'Val: {val_metrics["loss"]:.4f}, {val_metrics["acc"]:.2f}%'
        )

    print('Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

    with open(save_dir / 'train_history.json', 'w') as f:
        json.dump(train_history, f)
    with open(save_dir / 'val_history.json', 'w') as f:
        json.dump(val_history, f)

    return model, {'train': train_history, 'val': val_history}
