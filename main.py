
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.cross_validation import run_cross_validation
from src.final_model import train_final_model
from src.data_loader import resolve_h5_path


def require_run_dir(run_dir):
    if not run_dir:
        raise ValueError('This mode requires a completed cross-validation directory specified with --run_dir.')
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'Run directory does not exist: {run_dir}')
    return run_dir


def resolve_training_data(data_path=None):
    return resolve_h5_path(
        data_path or config.DEFAULT_TRAINING_DATA, purpose='training')


def resolve_recorded_training_data(run_dir, override=None):
    if override:
        return resolve_training_data(override)
    manifest_path = os.path.join(run_dir, 'run_manifest.json')
    if not os.path.isfile(manifest_path):
        print('Warning: run manifest not found; select a training file again.')
        return resolve_training_data()
    with open(manifest_path, 'r', encoding='utf-8-sig') as handle:
        recorded_path = json.load(handle).get('data_path')
    if not recorded_path:
        print('Warning: the run manifest has no training file; select one again.')
        return resolve_training_data()
    if not os.path.isabs(recorded_path):
        recorded_path = os.path.join(config.BASE_DIR, recorded_path)
    selected = resolve_h5_path(recorded_path, purpose='recorded training')
    print(f'Reusing the training file recorded by cross-validation: {selected}')
    return selected


def main():
    parser = argparse.ArgumentParser(description='CNN-LSTM training and evaluation workflow')
    parser.add_argument(
        '--mode', choices=['full', 'cv', 'final'], default='full',
        help=('full=cross-validation and final model training; cv=five-fold experiments; '
              'final=train the final full-data model'))
    parser.add_argument('--run_dir', help='Run directory for continuing an existing experiment')
    parser.add_argument(
        '--data_path',
        help='Training HDF5 file or directory; new runs default to interactive selection in data/training')
    parser.add_argument('--output_base', default=os.path.join(config.OUTPUT_DIR, 'cross_validation'))
    args = parser.parse_args()

    if args.mode == 'full':
        data_path = resolve_training_data(args.data_path)
        run_dir = run_cross_validation(data_path, args.output_base)
        train_final_model(data_path, run_dir)
        print(f'Training and evaluation workflow finished: {run_dir}')
        return

    if args.mode == 'cv':
        run_cross_validation(resolve_training_data(args.data_path), args.output_base)
        return

    run_dir = require_run_dir(args.run_dir)
    data_path = resolve_recorded_training_data(run_dir, args.data_path)
    train_final_model(data_path, run_dir)
    print(f'{args.mode} stage complete: {run_dir}')


if __name__ == '__main__':
    main()
