
import torch
import torch.nn as nn
from . import config


class MultiTaskHybridCNN_LSTM(nn.Module):
    def __init__(self, static_input_dim=3, use_lstm=True,
                 model_config=None, dropout_rate=None):
        super().__init__()
        self.static_input_dim = static_input_dim
        self.use_lstm = use_lstm
        self.use_cnn_residual = use_lstm
        architecture = dict(config.MODEL_CONFIG if model_config is None else model_config)
        dropout_rate = (config.SEARCH_REFERENCE_PARAMS['dropout_rate']
                        if dropout_rate is None else float(dropout_rate))
        conv1_filters = architecture['conv1_filters']
        conv2_filters = architecture['conv2_filters']
        kernel_size = architecture['kernel_size']
        lstm_hidden_size = architecture['lstm_hidden_size']
        lstm_num_layers = architecture['lstm_num_layers']
        dense_units = architecture['dense_units']
        self.architecture_config = architecture
        self.dropout_rate = dropout_rate


        self.cnn = nn.Sequential(
            nn.Conv1d(2, conv1_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )


        with torch.no_grad():
            test_input = torch.randn(1, 2, 250)
            test_output = self.cnn(test_input)
            cnn_out_features = test_output.size(1)
            cnn_out_length = test_output.size(2)


        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=cnn_out_features,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout_rate if lstm_num_layers > 1 else 0
            )

            sequence_output_dim = lstm_hidden_size + cnn_out_features
        else:
            self.lstm = None
            sequence_output_dim = cnn_out_features


        if static_input_dim > 0:
            self.static_branch = nn.Sequential(
                nn.Linear(static_input_dim, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            )
            static_output_dim = 32
        else:
            self.static_branch = None
            static_output_dim = 0


        self.shared_fc = nn.Sequential(
            nn.Linear(sequence_output_dim + static_output_dim, dense_units),
            nn.LayerNorm(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )


        self.reg_head = nn.Sequential(
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Linear(dense_units // 2, 3)
        )


        self.cls_head = nn.Sequential(
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Linear(dense_units // 2, 3),
            nn.Sigmoid()
        )


        self.cnn_out_features = cnn_out_features
        self.cnn_out_length = cnn_out_length
        self.lstm_hidden_size = lstm_hidden_size

    def forward(self, ts, static):

        ts = ts.permute(0, 2, 1)


        cnn_out = self.cnn(ts)

        if self.use_lstm:
            cnn_sequence = cnn_out.permute(0, 2, 1)
            lstm_out, _ = self.lstm(cnn_sequence)
            lstm_features = lstm_out[:, -1, :]
            cnn_global_features = cnn_out.mean(dim=2)
            sequence_features = torch.cat([lstm_features, cnn_global_features], dim=1)
        else:
            sequence_features = cnn_out.mean(dim=2)

        if self.static_branch is not None:
            static_out = self.static_branch(static)
            combined = torch.cat([sequence_features, static_out], dim=1)
        else:
            combined = sequence_features
        shared = self.shared_fc(combined)


        reg_output = self.reg_head(shared)
        cls_output = self.cls_head(shared)

        return reg_output, cls_output
