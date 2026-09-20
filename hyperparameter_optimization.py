
import json
import os

import numpy as np
import optuna
import torch

from . import config
from .data_loader import prepare_indexed_dataloaders
from .evaluation import (
    calculate_classification_metrics, calculate_regression_metrics, collect_predictions
)
from .models import MultiTaskHybridCNN_LSTM
from .train import train_model


def suggest_training_params(trial):
    settings = config.OPTUNA_CONFIG
    return config.make_training_params(
        lr=trial.suggest_float(
            'lr', settings['lr_low'], settings['lr_high'], log=True),
        weight_decay=trial.suggest_float(
            'weight_decay', settings['weight_decay_low'],
            settings['weight_decay_high'], log=True),
        dropout_rate=trial.suggest_float(
            'dropout_rate', settings['dropout_low'], settings['dropout_high']))


def composite_objective(reg_true, reg_pred, cls_true, cls_pred):
    regression = calculate_regression_metrics(reg_true, reg_pred)
    classification = calculate_classification_metrics(cls_true, cls_pred)
    macro_nrmse = float(np.nanmean(regression['nrmse']))
    macro_auc = float(np.nanmean(classification['auc']))
    if not np.isfinite(macro_nrmse) or not np.isfinite(macro_auc):
        raise optuna.TrialPruned('Validation data cannot produce finite Macro NRMSE and Macro ROC-AUC')
    settings = config.OPTUNA_CONFIG
    value = (settings['objective_nrmse_weight'] * macro_nrmse
             + settings['objective_auc_weight'] * (1.0 - macro_auc))
    return float(value), macro_nrmse, macro_auc


def optimize_training_params(X_ts, X_static, y_reg, y_cls, train_idx, val_idx,
                             device, output_dir, study_name, seed_offset=0):
    os.makedirs(output_dir, exist_ok=False)
    settings = config.OPTUNA_CONFIG
    sampler_seed = int(settings['sampler_seed'] + seed_offset)
    training_seed = int(settings['trial_training_seed'] + seed_offset)
    storage_path = os.path.abspath(os.path.join(output_dir, 'study.sqlite3'))
    storage_url = f"sqlite:///{storage_path.replace(os.sep, '/')}"
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        study_name=study_name, direction=settings['direction'], sampler=sampler,
        pruner=optuna.pruners.NopPruner(), storage=storage_url,
        load_if_exists=False)

    def objective(trial):

        config.set_seed(training_seed)
        training_params = suggest_training_params(trial)
        loaders, _, _, target_scaler = prepare_indexed_dataloaders(
            X_ts, X_static, y_reg, y_cls, train_idx, val_idx, val_idx,
            training_params['batch_size'], shuffle_seed=training_seed)
        train_loader, val_loader, _ = loaders
        model = MultiTaskHybridCNN_LSTM(
            static_input_dim=X_static.shape[1], use_lstm=True,
            model_config=config.MODEL_CONFIG,
            dropout_rate=training_params['dropout_rate'])
        checkpoint_path = os.path.join(output_dir, f'trial_{trial.number:04d}.pt')
        try:
            model, _ = train_model(
                train_loader, val_loader, device, checkpoint_path, model=model,
                checkpoint_metadata={
                    'role': 'Optuna internal validation only',
                    'study_name': study_name, 'trial_number': trial.number,
                    'train_indices': np.asarray(train_idx).tolist(),
                    'validation_indices': np.asarray(val_idx).tolist(),
                    'sampler_seed': sampler_seed, 'training_seed': training_seed},
                run_label=f'{study_name} | Trial {trial.number + 1}',
                training_params=training_params)
            reg_true, reg_pred, cls_true, cls_pred = collect_predictions(
                model, val_loader, device, target_scaler)
            value, macro_nrmse, macro_auc = composite_objective(
                reg_true, reg_pred, cls_true, cls_pred)
            saved = torch.load(checkpoint_path, map_location='cpu')
            trial.set_user_attr('best_epoch', int(saved['epoch']))
            trial.set_user_attr('macro_nrmse', macro_nrmse)
            trial.set_user_attr('macro_auc', macro_auc)
            trial.set_user_attr('training_seed', training_seed)
            return value
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    study.optimize(objective, n_trials=int(settings['n_trials']), gc_after_trial=True)
    best_params = config.make_training_params(**study.best_trial.params)
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
    summary = {
        'study_name': study_name,
        'direction': settings['direction'],
        'n_trials': int(settings['n_trials']),
        'sampler': 'TPESampler',
        'pruner': 'NopPruner',
        'optuna_version': optuna.__version__,
        'sampler_seed': sampler_seed,
        'trial_training_seed': training_seed,
        'search_space': {
            'lr': [settings['lr_low'], settings['lr_high'], 'log'],
            'weight_decay': [settings['weight_decay_low'],
                             settings['weight_decay_high'], 'log'],
            'dropout_rate': [settings['dropout_low'], settings['dropout_high'], 'linear'],
        },
        'fixed_training_params': dict(config.FIXED_TRAINING_PARAMS),
        'architecture_config': dict(config.MODEL_CONFIG),
        'objective': '0.5 * Macro NRMSE + 0.5 * (1 - Macro AUC)',
        'best_trial': int(study.best_trial.number),
        'best_value': float(study.best_value),
        'best_epoch': int(study.best_trial.user_attrs['best_epoch']),
        'best_params': best_params,
        'train_indices': np.asarray(train_idx).tolist(),
        'validation_indices': np.asarray(val_idx).tolist(),
    }
    with open(os.path.join(output_dir, 'best_params.json'),
              'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return best_params, summary
