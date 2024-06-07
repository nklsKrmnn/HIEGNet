from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.mlp import MLP
from src.utils.shape_calculation import calc_conv_output_size

SUBMODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "gcn": GCN,
    "mlp": MLP,
    "gcn_jk": GCNJumpingKnowledge,
    "gat_v2": GATv2
}


# Define the GCN model
class GnnCnnHybrid(nn.Module):
    def __init__(self,
                 hidden_channels,
                 image_size,
                 gnn_params,
                 input_channels=3,
                 kernel_size=3,
                 stride=3,
                 padding=0,
                 device=torch.device('cpu')):
        super(GnnCnnHybrid, self).__init__()
        self.device = device

        self.hidden_channels = hidden_channels
        self.in_channels = input_channels

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

        # Fully connection layer
        self.connention_fc = nn.Linear(self.hidden_channels[-1] * last_dim ** 2, gnn_params["input_dim"])

        # GNN layer
        self.gnn = SUBMODEL_NAME_MAPPING[gnn_params.pop("class_name")](**gnn_params)

    def forward(self, x, edge_index):
        # CNN
        x = self.cnn_encoder(x)

        # FC
        x = torch.flatten(x, start_dim=1)
        x = self.connention_fc(x)
        x = F.relu(x)

        # GNN
        x = self.gnn(x, edge_index)
        return F.softmax(x, dim=1)