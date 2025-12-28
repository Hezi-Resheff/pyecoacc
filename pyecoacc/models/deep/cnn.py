import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, Checkpoint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BehaviorCNNClassifier(nn.Module):
    def __init__(self, num_classes, sequence_length, conv_filters=[32, 64, 128, 256], 
                 fc_layers=[256, 512], kernel_size=5):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # XYZ 
        
        # 1. Dynamically build Convolutional Blocks
        for out_channels in conv_filters:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.4)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        self.flattened_size = self._get_flatten_size(sequence_length)

        # 2. Dynamically build Fully Connected Layers
        self.fc_blocks = nn.ModuleList()
        curr_features = self.flattened_size
        
        for hidden_units in fc_layers:
            self.fc_blocks.append(nn.Linear(curr_features, hidden_units))
            self.fc_blocks.append(nn.ReLU())
            self.fc_blocks.append(nn.Dropout(0.4))
            curr_features = hidden_units

        # Final Output Layer
        self.output_layer = nn.Linear(curr_features, num_classes)

    def _get_flatten_size(self, sequence_length):
        """Passes a dummy tensor through conv layers to find the flat shape."""
        with torch.no_grad():
            # (Batch, Channels, Length)
            dummy_input = torch.zeros(1, 3, sequence_length)
            for block in self.conv_blocks:
                dummy_input = block(dummy_input)
            return dummy_input.numel()

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = x.view(x.size(0), -1)

        for fc_layer in self.fc_blocks:
            x = fc_layer(x)

        return self.output_layer(x)



class CNNInputReshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes input data for CNN processing of accelerometer data.
    Transforms 2D input (n_samples, 3*sequence_length) into
    3D output (n_samples, 3, sequence_length) for CNN input.
    """

    def fit(self, X, y=None):
        return self

    @staticmethod
    def arrange_acc_for_cnn(X):
        sequence_len = X.shape[1] // 3
        return np.transpose(X.reshape(-1, sequence_len, 3), (0, 2, 1)).astype(np.float32)

    def transform(self, X):
        return self.arrange_acc_for_cnn(X)


def make_cnn_model(input_dim, num_behav, 
                   conv_filters=[32, 64, 128, 256], 
                   fc_layers=[256, 512], 
                   kernel_size=5,
                   max_epoch=200, lr=.001, l2=5e-4, verbose=1, validation=.2, patience=50, allow_cuda=True):
    
    net = NeuralNetClassifier(
        module=BehaviorCNNClassifier,
        module__conv_filters=conv_filters,
        module__fc_layers=fc_layers,
        module__kernel_size=kernel_size,  
        module__num_classes=num_behav,
        module__sequence_length=input_dim,
        criterion=nn.CrossEntropyLoss,
        optimizer__weight_decay=l2,
        
        max_epochs=max_epoch,
        lr=lr,
        optimizer=torch.optim.Adam,
        train_split=ValidSplit(validation) if validation is not None else validation,

        callbacks=[
            EarlyStopping(monitor='valid_acc', lower_is_better=False, patience=patience, load_best=True),
            Checkpoint(monitor='valid_acc_best')
        ],

        verbose=verbose,
        device='cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    )

    return Pipeline([
        ('reshape', CNNInputReshaper()),
        ('CNN', net)
    ])
