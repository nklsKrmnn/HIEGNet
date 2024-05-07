import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import cv2
import gc

from src.utils.shape_calculation import calc_conv_output_size


# Define the GCN model
class GnnCnnHybrid(nn.Module):
    def __init__(self, g_input_dim,
                 g_hidden_dim,
                 g_output_dim,
                 hidden_channels,
                 image_size,
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

        # Fully connected layer
        self.connention_fc = nn.Linear(self.hidden_channels[-1] * last_dim ** 2, g_input_dim)

        # GNN layer
        self.gconv1 = GCNConv(g_input_dim, g_hidden_dim)
        self.gconv2 = GCNConv(g_hidden_dim, g_hidden_dim)
        self.readout = nn.Linear(g_hidden_dim, g_output_dim)

    def forward(self, x, edge_index):

        input_x = x[0]
        x = []
        i = 0
        for x_i in input_x:
            x_i = cv2.imread(x_i)
            x_i = torch.tensor(x_i, dtype=torch.float)
            x_i = x_i.to(self.device).float().permute(2, 0, 1).unsqueeze(0)

            x_i = self.cnn_encoder(x_i)
            x_i = x_i.flatten(start_dim=1)
            x_i = self.connention_fc(x_i)

            x.append(x_i)

        x = torch.stack(x).squeeze(1)

        x = self.gconv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index)
        x = F.relu(x)
        x = self.readout(x)
        return F.softmax(x, dim=1)