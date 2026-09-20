
import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config
from .data_loader import (
    load_h5_data, prepare_indexed_dataloaders, resolve_h5_path, select_static_features,
    split_targets, joint_presence_labels
)
from .evaluation import (
    calculate_classification_metrics, calculate_regression_metrics, collect_predictions
)
from .models import MultiTaskHybridCNN_LSTM
from .hyperparameter_optimization import optimize_training_params
from .train import train_model


N_SPLITS = 5
INNER_VAL_SIZE = 0.20

EXPERIMENTS = [
    {'id': 'full_cnn_lstm', 'label': 'Full CNN-LSTM',
     'features': ['pH', 'Conductivity', 'T'], 'use_lstm': True},
    {'id': 'no_environmental_inputs', 'label': 'No environmental inputs',
     'features': [], 'use_lstm': True},
    {'id': 'cnn_only', 'label': 'CNN-only',
     'features': ['pH', 'Conductivity', 'T'], 'use_lstm': False},
]


def make_timestamped_run_dir(base_dir):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = os.path.join(base_dir, stamp)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f'{stamp}_{suffix:02d}')
        suffix += 1
    os.makedirs(candidate)
    for name in ('checkpoints', 'metrics', 'logs', 'scalers',
                 'optuna', 'final_model'):
        os.makedirs(os.path.join(candidate, name))
    return candidate


def scaler_state(scaler):
    if scaler is None:
        return None
    return {
        'mean_': scaler.mean_.tolist(), 'scale_': scaler.scale_.tolist(),
        'var_': scaler.var_.tolist(), 'n_features_in_': int(scaler.n_features_in_)
    }


def metric_rows(experiment, fold, reg_true, reg_pred, cls_true, cls_pred):
    reg = calculate_regression_metrics(reg_true, reg_pred)
    cls = calculate_classification_metrics(cls_true, cls_pred)
    rows = []
    for i, analyte in enumerate(config.reg_features):
        for metric in reg:
            rows.append({'experiment': experiment['id'], 'model': experiment['label'],
                         'fold': fold,
                         'task': 'Regression', 'analyte': analyte,
                         'metric': metric.upper(), 'value': reg[metric][i]})
    for i, analyte in enumerate(config.cls_features):
        for metric in cls:
            rows.append({'experiment': experiment['id'], 'model': experiment['label'],
                         'fold': fold,
                         'task': 'Classification', 'analyte': analyte,
                         'metric': metric.upper(), 'value': cls[metric][i]})
    return rows


def summarize_metrics(fold_metrics):
    macro_source = fold_metrics[fold_metrics['metric'].isin(['R2', 'NRMSE', 'AUC', 'F1'])]
    macro = (macro_source.groupby(['experiment', 'model', 'fold', 'task', 'metric'],
                                  as_index=False)['value'].mean())
    macro_summary = (macro.groupby(['experiment', 'model', 'task', 'metric'],
                                   as_index=False)['value']
                     .agg(mean='mean', sd='std', n_folds='count'))
    per_analyte_source = fold_metrics[fold_metrics['metric'] != 'NRMSE']
    per_analyte = (per_analyte_source.groupby(
        ['experiment', 'model', 'task', 'analyte', 'metric'], as_index=False)['value']
        .agg(mean='mean', sd='std', n_folds='count'))
    return macro, macro_summary, per_analyte


def save_main_table(macro_summary, path):
    primary = macro_summary[
        macro_summary['metric'].isin(['R2', 'NRMSE', 'AUC', 'F1'])
    ].copy()
    primary['mean_sd'] = primary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['sd']:.4f}", axis=1)
    table = primary.pivot(index='model', columns='metric', values='mean_sd').reset_index()
    ordered = ['model', 'R2', 'NRMSE', 'AUC', 'F1']
    table[[c for c in ordered if c in table.columns]].to_csv(path, index=False, encoding="utf-8-sig")


def run_cross_validation(data_path, output_base):
    data_path = resolve_h5_path(data_path, purpose='training')
    run_dir = make_timestamped_run_dir(output_base)
    device = config.get_device()
    X_ts, X_static_all, y = load_h5_data(data_path)
    y_reg, y_cls = split_targets(y)
    strata = joint_presence_labels(y_cls)

    counts = dict(zip(*np.unique(strata, return_counts=True)))
    if min(counts.values()) < N_SPLITS:
        raise ValueError(f'The smallest joint-label stratum has {min(counts.values())} samples; five-fold stratification is not possible.')

    manifest = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'data_path': os.path.abspath(data_path), 'n_samples': int(len(y)),
        'n_splits': N_SPLITS, 'outer_seed': config.RANDOM_STATE,
        'training_seed_base': config.SEED, 'inner_validation_fraction': INNER_VAL_SIZE,
        'model_architecture': dict(config.MODEL_CONFIG),
        'fixed_training_params': dict(config.FIXED_TRAINING_PARAMS),
        'optuna_config': dict(config.OPTUNA_CONFIG),
        'optuna_scope': (
            'independent study inside each outer development fold; outer test fold unseen'),
        'optuna_objective': '0.5 * Macro NRMSE + 0.5 * (1 - Macro AUC)',
        'reference_model_hyperparameters': (
            'all ablation/reference models inherit the corresponding outer-fold '
            'Full CNN-LSTM best training parameters'),
        'stratification': 'joint Zn/Cd/Pb presence label',
        'nrmse_definition': 'RMSE divided by observed range in the held-out fold',
        'regression_target_scaling': (
            'Zn/Cd/Pb targets standardized using training-fold mean and SD; '
            'predictions inverse-transformed before all reported metrics'),
        'sequence_fusion': 'LSTM last-step features concatenated with CNN global-average features',
        'classification_threshold': 0.5,
        'primary_manuscript_metrics': ['Macro R2', 'Macro NRMSE', 'Macro ROC-AUC', 'Macro F1 score'],
        'experiments': EXPERIMENTS, 'joint_label_counts': {str(k): int(v) for k, v in counts.items()}
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    all_rows, prediction_frames = [], []
    best_params_rows = []
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_splits = []
    for fold, (development_idx, test_idx) in enumerate(
            outer.split(X_ts, strata), start=1):
        train_idx, val_idx = train_test_split(
            development_idx, test_size=INNER_VAL_SIZE, random_state=config.RANDOM_STATE + fold,
            stratify=strata[development_idx]
        )
        fold_splits.append({
            'fold': fold,
            'outer_development_indices': development_idx.tolist(),
            'train_indices': train_idx.tolist(),
            'validation_indices': val_idx.tolist(),
            'outer_test_indices': test_idx.tolist(),
            'inner_split_seed': config.RANDOM_STATE + fold,
        })
    with open(os.path.join(run_dir, 'data_splits.json'),
              'w', encoding='utf-8') as handle:
        json.dump(fold_splits, handle, ensure_ascii=False, indent=2)

    for split in fold_splits:
        fold = split['fold']
        train_idx = np.asarray(split['train_indices'], dtype=int)
        val_idx = np.asarray(split['validation_indices'], dtype=int)
        test_idx = np.asarray(split['outer_test_indices'], dtype=int)
        full_static, _ = select_static_features(
            X_static_all)
        fold_params, optuna_summary = optimize_training_params(
            X_ts, full_static, y_reg, y_cls, train_idx, val_idx, device,
            os.path.join(run_dir, 'optuna', f'fold_{fold}'),
            study_name=f'outer_fold_{fold}_full_cnn_lstm', seed_offset=fold)
        best_params_rows.append({
            'fold': fold,
            **fold_params,
            'best_trial': optuna_summary['best_trial'],
            'best_objective': optuna_summary['best_value'],
            'optuna_best_epoch': optuna_summary['best_epoch'],
        })
        for experiment_position, experiment in enumerate(EXPERIMENTS, start=1):
            config.set_seed(config.SEED + fold)
            if experiment['features']:
                X_static, selected_indices = select_static_features(
                    X_static_all)
            else:
                X_static = np.empty((len(X_static_all), 0), dtype=X_static_all.dtype)
                selected_indices = []

            loaders, scaler_ts, scaler_static, scaler_target = prepare_indexed_dataloaders(
                X_ts, X_static, y_reg, y_cls, train_idx, val_idx, test_idx,
                fold_params['batch_size'], shuffle_seed=config.SEED + fold)
            train_loader, val_loader, test_loader = loaders
            model = MultiTaskHybridCNN_LSTM(
                static_input_dim=X_static.shape[1], use_lstm=experiment['use_lstm'],
                model_config=config.MODEL_CONFIG,
                dropout_rate=fold_params['dropout_rate'])
            checkpoint_path = os.path.join(
                run_dir, 'checkpoints', f"{experiment['id']}_fold_{fold}.pt")
            metadata = {
                'experiment': experiment, 'fold': fold,
                'training_params_source': 'Full CNN-LSTM Optuna study in this outer fold',
                'training_params': dict(fold_params),
                'training_seed': config.SEED + fold,
                'optuna_best_trial': optuna_summary['best_trial'],
                'optuna_best_objective': optuna_summary['best_value'],
                'train_indices': train_idx.tolist(), 'validation_indices': val_idx.tolist(),
                'test_indices': test_idx.tolist(), 'selected_static_indices': selected_indices,
                'time_series_scaler': scaler_state(scaler_ts),
                'static_scaler': scaler_state(scaler_static),
                'regression_target_scaler': scaler_state(scaler_target)
            }
            model, history = train_model(
                train_loader, val_loader, device, checkpoint_path,
                model=model, checkpoint_metadata=metadata,
                run_label=(
                    f"CV {(fold - 1) * len(EXPERIMENTS) + experiment_position}/"
                    f"{N_SPLITS * len(EXPERIMENTS)} | {experiment['label']} | "
                    f"Fold {fold}/{N_SPLITS}"),
                training_params=fold_params)
            pd.DataFrame(history).to_csv(
                os.path.join(run_dir, 'logs', f"{experiment['id']}_fold_{fold}.csv"), index=False)
            joblib.dump({'time_series': scaler_ts, 'static': scaler_static,
                         'regression_target': scaler_target}, os.path.join(
                run_dir, 'scalers', f"{experiment['id']}_fold_{fold}.joblib"))

            reg_true, reg_pred, cls_true, cls_pred = collect_predictions(
                model, test_loader, device, scaler_target)
            all_rows.extend(metric_rows(
                experiment, fold, reg_true, reg_pred, cls_true, cls_pred))
            prediction_frames.append(pd.DataFrame({
                'experiment': experiment['id'], 'fold': fold,
                'sample_index': test_idx,
                **{f'{name}_true': reg_true[:, i] for i, name in enumerate(config.reg_features)},
                **{f'{name}_pred': reg_pred[:, i] for i, name in enumerate(config.reg_features)},
                **{f'{name}_true': cls_true[:, i] for i, name in enumerate(config.cls_features)},
                **{f'{name}_prob': cls_pred[:, i] for i, name in enumerate(config.cls_features)}
            }))

    fold_metrics = pd.DataFrame(all_rows)
    macro_folds, macro_summary, per_analyte = summarize_metrics(fold_metrics)
    metrics_dir = os.path.join(run_dir, 'metrics')
    pd.DataFrame(best_params_rows).to_csv(
        os.path.join(run_dir, 'optuna', 'best_params_by_outer_fold.csv'), index=False)
    fold_metrics[fold_metrics['metric'] != 'NRMSE'].to_csv(
        os.path.join(metrics_dir, 'all_fold_metrics_long.csv'), index=False)
    macro_folds.to_csv(os.path.join(metrics_dir, 'macro_metrics_by_fold.csv'), index=False)
    macro_summary.to_csv(os.path.join(metrics_dir, 'macro_metrics_mean_sd.csv'), index=False)
    per_analyte.to_csv(os.path.join(metrics_dir, 'supplementary_per_analyte_mean_sd.csv'), index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(os.path.join(metrics_dir, 'out_of_fold_predictions.csv'), index=False)
    save_main_table(macro_summary, os.path.join(metrics_dir, 'main_table_mean_sd.csv'))
    pointer_path = os.path.join(output_base, 'latest_completed_run.txt')
    with open(pointer_path, 'w', encoding='utf-8') as handle:
        handle.write(os.path.abspath(run_dir))
    print(f'Cross-validation complete. Results saved to: {run_dir}')
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Five-fold joint-label stratified cross-validation and ablation experiments')
    parser.add_argument('--data_path', default=config.DEFAULT_TRAINING_DATA,
                        help='Training HDF5 file or directory; select a file by number when multiple files are available')
    parser.add_argument('--output_base', default=os.path.join(config.OUTPUT_DIR, 'cross_validation'))
    args = parser.parse_args()
    run_cross_validation(args.data_path, args.output_base)


if __name__ == '__main__':
    main()
