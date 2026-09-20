
import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def calculate_regression_metrics(true, pred):
    names = ('rmse', 'nrmse', 'mae', 'r2')
    metrics = {name: [] for name in names}
    for i in range(pred.shape[1]):
        true_feat = true[:, i]
        pred_feat = pred[:, i]
        rmse = np.sqrt(mean_squared_error(true_feat, pred_feat))
        observed_range = np.max(true_feat) - np.min(true_feat)
        metrics['rmse'].append(rmse)
        metrics['nrmse'].append(rmse / observed_range if observed_range > 0 else np.nan)
        metrics['mae'].append(mean_absolute_error(true_feat, pred_feat))
        metrics['r2'].append(r2_score(true_feat, pred_feat))
    return metrics


def calculate_classification_metrics(true, pred):
    names = ('accuracy', 'precision', 'recall', 'f1', 'auc')
    metrics = {name: [] for name in names}
    for i in range(pred.shape[1]):
        pred_labels = (pred[:, i] > 0.5).astype(int)
        true_labels = true[:, i]
        probabilities = np.clip(pred[:, i], 1e-7, 1 - 1e-7)
        metrics['accuracy'].append(accuracy_score(true_labels, pred_labels))
        metrics['precision'].append(precision_score(true_labels, pred_labels, zero_division=0))
        metrics['recall'].append(recall_score(true_labels, pred_labels, zero_division=0))
        metrics['f1'].append(f1_score(true_labels, pred_labels, zero_division=0))
        metrics['auc'].append(
            roc_auc_score(true_labels, probabilities)
            if np.unique(true_labels).size == 2 else np.nan
        )
    return metrics


def collect_predictions(model, loader, device, target_scaler):
    model.eval()
    reg_pred, reg_true, cls_pred, cls_true = [], [], [], []
    with torch.no_grad():
        for ts, static, y_reg, y_cls in loader:
            pred_reg, pred_cls = model(ts.to(device), static.to(device))
            reg_pred.append(pred_reg.cpu().numpy())
            cls_pred.append(pred_cls.cpu().numpy())
            reg_true.append(y_reg.numpy())
            cls_true.append(y_cls.numpy())
    reg_true, reg_pred, cls_true, cls_pred = tuple(
        np.concatenate(values) for values in (reg_true, reg_pred, cls_true, cls_pred))
    return (target_scaler.inverse_transform(reg_true),
            target_scaler.inverse_transform(reg_pred), cls_true, cls_pred)
