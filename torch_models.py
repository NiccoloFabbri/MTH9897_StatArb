"""
Gabo Bernardino, Niccolò Fabbri

Pytorch models for Systematic Trading project
"""


import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

torch.manual_seed(1)


class RNC(nn.Module):
    r"""
    Rete Neurale Convoluzionale, aka Convolutional Neural Network
    Has 2 convolutions:
    * The first one slides the filters horizontally on each row, aggregating
    all the contracts related to a single commodity;
    * The second one slides the filters vertically, aggregating the constructed
    features over the time dimension (by default, over one week)
    These are followed by two fully connected layers of size `hidden_size`

    The input should have a single channel, with a `seq_length` x `n_assets*15`
    time series of residual returns for each of the contracts
    """
    def __init__(self,
                 activation,
                 seq_length: int = 21,
                 out_channels1: int = 2,
                 out_channels2: int = 2,
                 n_assets: int = 2,
                 hidden_size: int = 32,
                 time_kernel_size: Tuple[int, int] = (5, 1),
                 time_stride: int = 1,
                 padding: int = 0):
        super().__init__()
        self._activation = activation
        self.n_assets = n_assets
        self.n_contracts = 15  # number of contracts per commodity
        self.kernel_comm = (1, self.n_contracts)
        self.kernel_time = time_kernel_size
        self.in_channels = 1
        # first convolution - aggregate contracts for same comm
        self.conv1 = nn.Conv2d(
            self.in_channels, out_channels1,
            kernel_size=self.kernel_comm, stride=self.kernel_comm,
            padding=padding
        )
        # second convolution - aggregate along time dimension
        self.conv2 = nn.Conv2d(
            out_channels1, out_channels2,
            kernel_size=self.kernel_time, stride=time_stride,
            padding=padding
        )
        self.fc1 = nn.Linear(
            (seq_length-time_kernel_size[0]+1) * out_channels2 * self.n_assets,
            hidden_size
        )
        self.fc2 = nn.Linear(hidden_size, self.n_assets * self.n_contracts)

    def forward(self, x):
        x = self._activation(self.conv1(x))
        x = self._activation(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self._activation(self.fc1(x))
        x = self.fc2(x)
        return x
