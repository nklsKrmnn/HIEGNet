import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout=0.0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # Add the first layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_dims[0]))

        # Add subsequent layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i]))

        # Add the output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_rate = dropout

    def forward(self, x, _):
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = layer_norm(x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.output_layer(x)
        return x


