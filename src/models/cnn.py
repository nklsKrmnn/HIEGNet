from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.mlp import MLP
from src.utils.shape_calculation import calc_conv_output_size

class CNN(nn.Module):
    def __init__(self,
                 hidden_channels,
                 image_size,
                 mlp_params,
                 input_channels=3,
                 kernel_size=3,
                 stride=3,
                 padding=0,
                 cnn_softmax_function="softmax",
                 device=torch.device('cpu')):
        super(CNN, self).__init__()
        self.device = device

        self.hidden_channels = hidden_channels
        self.in_channels = input_channels
        self.softmax_function = cnn_softmax_function

        last_dim = image_size
        for _ in self.hidden_channels:
            last_dim = calc_conv_output_size(last_dim, kernel_size, stride, padding)

        # CNN layers
        modules = []
        for h_channels in self.hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=h_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
                    nn.BatchNorm2d(h_channels),
                    nn.LeakyReLU()
                )
            )

            self.in_channels = h_channels

        self.cnn_encoder = nn.Sequential(*modules)

        # Set input dim for MLP
        mlp_params["input_dim"] = self.hidden_channels[-1] * last_dim ** 2

        # GNN layer
        mlp_params = mlp_params.copy()
        mlp_params["mlp_softmax_function"] = 'none'
        self.mlp = MLP(**mlp_params)

    def forward(self, x):
        # CNN
        x = self.cnn_encoder(x)

        # FC
        x = torch.flatten(x, start_dim=1)
        x = F.relu(x)

        # MLP
        x = self.mlp(x)

        # Apply softmax if needed
        if self.softmax_function == "softmax":
            output = F.softmax(x, dim=1)
        elif self.softmax_function == "log_softmax":
            output = F.log_softmax(x, dim=1)
        elif self.softmax_function == "none":
            output = x
        else:
            raise ValueError(f"Unknown softmax function: {self.softmax_function}")

        return output

class CBR(nn.Module):
    def __init__(self,
                 hidden_channels,
                 image_size,
                 mlp_params,
                 input_channels=3,
                 kernel_size=3,
                 stride=3,
                 padding=0,
                 cnn_softmax_function="softmax",
                 device=torch.device('cpu')):
        super(CNN, self).__init__()
        self.device = device

        self.hidden_channels = hidden_channels
        self.in_channels = input_channels
        self.softmax_function = cnn_softmax_function

        last_dim = image_size
        for _ in self.hidden_channels:
            last_dim = calc_conv_output_size(last_dim, kernel_size, stride, padding)

        # CNN layers
        modules = []
        for h_channels in self.hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels,
                              out_channels=h_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
                    nn.BatchNorm2d(h_channels),
                    nn.ReLU()
                )
            )
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))

            self.in_channels = h_channels

        self.cnn_encoder = nn.Sequential(*modules)

        # MLP layer
        mlp_params = mlp_params.copy()
        mlp_params["mlp_softmax_function"] = 'none'
        self.mlp = MLP(**mlp_params)

    def forward(self, x):
        # CNN
        x = self.cnn_encoder(x)

        # FC
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = F.relu(x)

        # MLP
        x = self.mlp(x)

        # Apply softmax if needed
        if self.softmax_function == "softmax":
            output = F.softmax(x, dim=1)
        elif self.softmax_function == "log_softmax":
            output = F.log_softmax(x, dim=1)
        elif self.softmax_function == "none":
            output = x
        else:
            raise ValueError(f"Unknown softmax function: {self.softmax_function}")

        return output