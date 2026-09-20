
import os
import random


import torch
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training")
DEFAULT_TRAINING_DATA = TRAINING_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


SEED = 74

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


reg_features = ["Zn Concentration", "Cd Concentration", "Pb Concentration"]
cls_features = ["Zn Exist", "Cd Exist", "Pb Exist"]

static_feature_names = ["pH", "Conductivity", "T"]


MODEL_CONFIG = {
    'conv1_filters': 64,
    'conv2_filters': 128,
    'kernel_size': 3,
    'lstm_hidden_size': 256,
    'lstm_num_layers': 3,
    'dense_units': 512,
}


FIXED_TRAINING_PARAMS = {
    'batch_size': 16,
    'reg_weight': 0.5,
    'cls_weight': 0.5,
    'max_grad_norm': 3.5,
}


SEARCH_REFERENCE_PARAMS = {
    'lr': 8.35e-5,
    'weight_decay': 2.93e-6,
    'dropout_rate': 0.108,
}

OPTUNA_CONFIG = {
    'n_trials': 30,
    'direction': 'minimize',
    'sampler_seed': 20260829,

    'trial_training_seed': SEED,
    'lr_low': 2.0e-05,
    'lr_high': 3.0e-04,
    'weight_decay_low': 1.0e-07,
    'weight_decay_high': 1.0e-04,
    'dropout_low': 0.0,
    'dropout_high': 0.3,
    'objective_nrmse_weight': 0.5,
    'objective_auc_weight': 0.5,
}




def make_training_params(lr=None, weight_decay=None, dropout_rate=None):
    params = dict(FIXED_TRAINING_PARAMS)
    params.update({
        'lr': SEARCH_REFERENCE_PARAMS['lr'] if lr is None else float(lr),
        'weight_decay': (
            SEARCH_REFERENCE_PARAMS['weight_decay']
            if weight_decay is None else float(weight_decay)),
        'dropout_rate': (
            SEARCH_REFERENCE_PARAMS['dropout_rate']
            if dropout_rate is None else float(dropout_rate)),
    })
    return params


MAX_EPOCHS = 20000
PATIENCE = 100


RANDOM_STATE = 42


