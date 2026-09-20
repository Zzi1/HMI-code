
import torch
import torch.nn as nn
import torch.optim as optim
from . import config
from .models import MultiTaskHybridCNN_LSTM


def setup_training(model, device, training_params):
    reg_weight = training_params['reg_weight']
    cls_weight = training_params['cls_weight']

    reg_criterion = nn.HuberLoss()
    cls_criterion = nn.BCELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params['lr'],
        weight_decay=training_params['weight_decay']
    )

    return reg_criterion, cls_criterion, optimizer, reg_weight, cls_weight


def train_one_epoch(model, train_loader, optimizer, reg_criterion, cls_criterion,
                    reg_weight, cls_weight, max_grad_norm, device):
    model.train()
    train_total_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_cls_loss = 0.0
    train_correct = 0
    train_total = 0

    for ts_batch, static_batch, reg_batch, cls_batch in train_loader:
        ts_batch = ts_batch.to(device)
        static_batch = static_batch.to(device)
        reg_batch = reg_batch.to(device)
        cls_batch = cls_batch.to(device)

        optimizer.zero_grad()
        reg_pred, cls_pred = model(ts_batch, static_batch)

        reg_loss = reg_criterion(reg_pred, reg_batch) * reg_weight
        cls_loss = cls_criterion(cls_pred, cls_batch) * cls_weight
        total_loss = reg_loss + cls_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        train_total_loss += total_loss.item() * ts_batch.size(0)
        epoch_reg_loss += reg_loss.item() * ts_batch.size(0)
        epoch_cls_loss += cls_loss.item() * ts_batch.size(0)

        predicted = (cls_pred > 0.5).float()
        train_correct += (predicted == cls_batch).sum().item()
        train_total += cls_batch.numel()

    train_loss = train_total_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return train_loss, epoch_reg_loss / len(train_loader.dataset),\
        epoch_cls_loss / len(train_loader.dataset), train_accuracy


def validate(model, val_loader, reg_criterion, cls_criterion,
             reg_weight, cls_weight, device):
    model.eval()
    val_total_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for ts_batch, static_batch, reg_batch, cls_batch in val_loader:
            ts_batch = ts_batch.to(device)
            static_batch = static_batch.to(device)
            reg_batch = reg_batch.to(device)
            cls_batch = cls_batch.to(device)

            reg_pred, cls_pred = model(ts_batch, static_batch)
            reg_loss = reg_criterion(reg_pred, reg_batch) * reg_weight
            cls_loss = cls_criterion(cls_pred, cls_batch) * cls_weight
            val_total_loss += (reg_loss + cls_loss).item() * ts_batch.size(0)

            predicted = (cls_pred > 0.5).float()
            val_correct += (predicted == cls_batch).sum().item()
            val_total += cls_batch.numel()

    val_loss = val_total_loss / len(val_loader.dataset)
    val_accuracy = 100.0 * val_correct / val_total

    return val_loss, val_accuracy


def train_model(train_loader, val_loader, device, model_save_path,
                model=None, checkpoint_metadata=None, run_label=None,
                training_params=None):
    training_params = dict(training_params or config.make_training_params())
    model = (model or MultiTaskHybridCNN_LSTM(
        dropout_rate=training_params['dropout_rate'])).to(device)
    reg_criterion, cls_criterion, optimizer, reg_weight, cls_weight =\
        setup_training(model, device, training_params)

    max_grad_norm = training_params['max_grad_norm']
    max_epochs = config.MAX_EPOCHS
    patience = config.PATIENCE

    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0

    history = {
        'train_total_losses': [],
        'val_total_losses': [],
        'reg_losses': [],
        'cls_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }

    prefix = f"[{run_label}] " if run_label else ""
    print(f"\n{prefix}Starting training...")
    print(f"{prefix}Model device: {next(model.parameters()).device}")

    for epoch in range(max_epochs):
        train_loss, reg_loss, cls_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, reg_criterion, cls_criterion,
            reg_weight, cls_weight, max_grad_norm, device
        )
        val_loss, val_acc = validate(
            model, val_loader, reg_criterion, cls_criterion,
            reg_weight, cls_weight, device
        )

        history['train_total_losses'].append(train_loss)
        history['val_total_losses'].append(val_loss)
        history['reg_losses'].append(reg_loss)
        history['cls_losses'].append(cls_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)

        if epoch % 500 == 0:
            print(f"{prefix}Epoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            print(f"{prefix}Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'checkpoint_version': 2,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': val_acc,
                'history': history,
                'training_params': dict(training_params),
                'architecture_config': dict(model.architecture_config),
                'model_config': {
                    'class_name': model.__class__.__name__,
                    'static_input_dim': getattr(model, 'static_input_dim', 3),
                    'use_lstm': getattr(model, 'use_lstm', True),
                    'use_cnn_residual': getattr(model, 'use_cnn_residual', False)
                },
                'metadata': checkpoint_metadata or {}
            }
            torch.save(checkpoint, model_save_path)
            best_val_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'{prefix}Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.4f}, '
                      f'Validation accuracy at best loss: {best_val_acc:.2f}%')
                break

    print(f"{prefix}Training complete after {epoch + 1} epochs, "
          f"Best validation loss: {best_val_loss:.4f}, "
          f"Validation accuracy at best loss: {best_val_acc:.2f}%")

    best_checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    return model, history


def train_model_fixed_epochs(train_loader, device, model_save_path, num_epochs,
                             model=None, checkpoint_metadata=None,
                             training_params=None):
    training_params = dict(training_params or config.make_training_params())
    model = (model or MultiTaskHybridCNN_LSTM(
        dropout_rate=training_params['dropout_rate'])).to(device)
    reg_criterion, cls_criterion, optimizer, reg_weight, cls_weight =\
        setup_training(model, device, training_params)
    max_grad_norm = training_params['max_grad_norm']
    history = {
        'train_total_losses': [], 'reg_losses': [], 'cls_losses': [],
        'train_accuracies': []
    }
    prefix = "[Final full-data model] "
    print(f"\n{prefix}Starting full-data training for {num_epochs} epochs...")
    print(f"{prefix}Model device: {next(model.parameters()).device}")
    for epoch in range(num_epochs):
        train_loss, reg_loss, cls_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, reg_criterion, cls_criterion,
            reg_weight, cls_weight, max_grad_norm, device)
        history['train_total_losses'].append(train_loss)
        history['reg_losses'].append(reg_loss)
        history['cls_losses'].append(cls_loss)
        history['train_accuracies'].append(train_acc)
        if epoch % 500 == 0 or epoch + 1 == num_epochs:
            print(f"{prefix}Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%")

    checkpoint = {
        'checkpoint_version': 2,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'training_params': dict(training_params),
        'architecture_config': dict(model.architecture_config),
        'model_config': {
            'class_name': model.__class__.__name__,
            'static_input_dim': getattr(model, 'static_input_dim', 3),
            'use_lstm': getattr(model, 'use_lstm', True),
            'use_cnn_residual': getattr(model, 'use_cnn_residual', False)
        },
        'metadata': checkpoint_metadata or {}
    }
    torch.save(checkpoint, model_save_path)
    return model, history
