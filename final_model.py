
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from . import config
from .data_loader import (
    MultiTaskDataset, load_h5_data, select_static_features,
    split_targets, standardize_time_series, to_tensor, joint_presence_labels
)
from .models import MultiTaskHybridCNN_LSTM
from .hyperparameter_optimization import optimize_training_params
from .train import train_model_fixed_epochs


FINAL_OPTUNA_VAL_SIZE = 0.20


def prepare_full_dataset(data_path):
    X_ts, X_static_all, y = load_h5_data(data_path)
    y_reg, y_cls = split_targets(y)
    X_static, selected_indices = select_static_features(
        X_static_all)
    scaler_ts, X_ts = standardize_time_series(X_ts)
    scaler_static = StandardScaler().fit(X_static)
    scaler_target = StandardScaler().fit(y_reg)
    X_static = scaler_static.transform(X_static)
    y_reg = scaler_target.transform(y_reg)
    dataset = MultiTaskDataset(
        to_tensor(X_ts, torch.device('cpu')),
        to_tensor(X_static, torch.device('cpu')),
        to_tensor(y_reg, torch.device('cpu')),
        to_tensor(y_cls, torch.device('cpu')))
    return dataset, scaler_ts, scaler_static, scaler_target, selected_indices


def train_final_model(data_path, run_dir):
    final_dir = os.path.join(run_dir, 'final_model')
    os.makedirs(final_dir, exist_ok=True)
    checkpoint_path = os.path.join(final_dir, 'final_model.pt')
    if os.path.exists(checkpoint_path):
        raise FileExistsError(
            f'Final model already exists: {checkpoint_path}. Use a new CV run directory to avoid overwriting it.')
    device = config.get_device()
    X_ts, X_static_all, y = load_h5_data(data_path)
    y_reg, y_cls = split_targets(y)
    X_static, _ = select_static_features(
        X_static_all)
    all_indices = np.arange(len(y))
    strata = joint_presence_labels(y_cls)
    tune_train_idx, tune_val_idx = train_test_split(
        all_indices, test_size=FINAL_OPTUNA_VAL_SIZE,
        random_state=config.RANDOM_STATE + 1000, stratify=strata)
    final_params, optuna_summary = optimize_training_params(
        X_ts, X_static, y_reg, y_cls, tune_train_idx, tune_val_idx, device,
        os.path.join(final_dir, 'optuna'), study_name='final_full_data_cnn_lstm',
        seed_offset=1000)
    final_epochs = max(1, int(optuna_summary['best_epoch']))

    config.set_seed(config.SEED + 1000)
    dataset, scaler_ts, scaler_static, scaler_target, selected_indices = prepare_full_dataset(data_path)
    final_loader_generator = torch.Generator().manual_seed(config.SEED + 1000)
    loader = DataLoader(
        dataset, batch_size=final_params['batch_size'], shuffle=True,
        generator=final_loader_generator)
    model = MultiTaskHybridCNN_LSTM(
        static_input_dim=3, use_lstm=True, model_config=config.MODEL_CONFIG,
        dropout_rate=final_params['dropout_rate'])
    metadata = {
        'role': 'final full-data trained model; not for performance estimation',
        'data_path': os.path.abspath(data_path), 'n_samples': len(dataset),
        'final_epochs': final_epochs,
        'epoch_selection': 'best epoch from the final full-data internal Optuna split',
        'training_params': dict(final_params),
        'training_seed': config.SEED + 1000,
        'architecture_config': dict(config.MODEL_CONFIG),
        'optuna_summary': optuna_summary,
        'performance_note': (
            'This full-data model has no independent test score; manuscript performance '
            'must come from outer-fold held-out predictions.'),
        'selected_static_features': list(config.static_feature_names),
        'selected_static_indices': selected_indices,
        'regression_target_scaling': (
            'standardized using full-data mean and SD; inference outputs require inverse transform'),
        'time_series_scaler': {
            'mean_': scaler_ts.mean_.tolist(), 'scale_': scaler_ts.scale_.tolist(),
            'var_': scaler_ts.var_.tolist()
        },
        'static_scaler': {
            'mean_': scaler_static.mean_.tolist(), 'scale_': scaler_static.scale_.tolist(),
            'var_': scaler_static.var_.tolist()
        },
        'regression_target_scaler': {
            'mean_': scaler_target.mean_.tolist(), 'scale_': scaler_target.scale_.tolist(),
            'var_': scaler_target.var_.tolist()
        }
    }
    model, history = train_model_fixed_epochs(
        loader, device, checkpoint_path, final_epochs,
        model=model, checkpoint_metadata=metadata, training_params=final_params)
    scaler_path = os.path.join(final_dir, 'final_scalers.joblib')
    joblib.dump({'time_series': scaler_ts, 'static': scaler_static,
                 'regression_target': scaler_target}, scaler_path)
    pd.DataFrame(history).to_csv(
        os.path.join(final_dir, 'training_history.csv'), index=False)
    with open(os.path.join(final_dir, 'final_model_manifest.json'),
              'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return model, dataset, device


