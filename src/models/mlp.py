from torch import nn
import torch.nn.functional as F

from src.models.model_utils import init_norm_layer

class MLP(nn.Module):

    def __init__(self, input_dim,
                 hidden_dims: list[int],
                 output_dim,
                 dropout=0.5,
                 norm: str = None):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout_rate = dropout
        self.norm = norm

        # First GAT layer
        self.fc_layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            init_norm_layer(self.norm)(hidden_dims[0])
        ))

        # Intermediate GAT and FC layers
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                init_norm_layer(self.norm)(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    def forward(self, x, _=None):

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)

        x = self.output_layer(x)

        return F.log_softmax(x, dim=1)


