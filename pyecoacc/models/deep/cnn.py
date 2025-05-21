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
    def __init__(self, num_classes, sequence_length):
        super().__init__()

        # Input shape: (batch_size, 3 ACC axes XYZ, sequence_length = sampling rate [Hz] * duration [s])

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 4th convolutional layer
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * (sequence_length // 16), 512)
        self.fc2 = nn.Linear(512, num_classes)

        # to be used everywhere
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Convolutional layers with norm ReLU and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def arrange_acc_for_cnn(X):
    sequence_len = X.shape[1] // 3
    return np.transpose(X.reshape(-1, sequence_len, 3), (0, 2, 1)).astype(np.float32)


class CNNInputReshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes input data for CNN processing of accelerometer data.
    Transforms 2D input (n_samples, 3*sequence_length) into
    3D output (n_samples, 3, sequence_length) for CNN input.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return arrange_acc_for_cnn(X)


def make_cnn_model(input_dim, num_behav, max_epoch=200, lr=.001, l2=5e-4, verbose=1, validation=.2, patience=50):
    net = NeuralNetClassifier(
        module=BehaviorCNNClassifier,
        criterion=nn.CrossEntropyLoss,
        optimizer__weight_decay=l2,

        module__num_classes=num_behav,
        module__sequence_length=input_dim,

        max_epochs=max_epoch,
        lr=lr,
        optimizer=torch.optim.Adam,
        train_split=ValidSplit(validation) if validation is not None else validation,

        callbacks=[
            EarlyStopping(monitor='valid_acc', lower_is_better=False, patience=patience, load_best=True),
            Checkpoint(monitor='valid_acc_best')
        ],

        verbose=verbose,
        device='cpu'
    )

    return Pipeline([
        ('reshape', CNNInputReshaper()),
        ('CNN', net)
    ])
