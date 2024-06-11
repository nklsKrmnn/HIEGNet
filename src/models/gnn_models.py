import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims: list[int], output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
        self.readout = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.readout(x)
        return F.softmax(x, dim=1)


class GCN2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(GCN2, self).__init__()
        hidden_dim = hidden_dims[0]
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.relu(self.conv3(x, edge_index))
        x = self.readout(x)
        return F.softmax(x, dim=1)


class GCNJumpingKnowledge2(nn.Module):
    def __init__(self, input_dim, hidden_dims: list[int], output_dim, dropout=0.5):
        super(GCNJumpingKnowledge2, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
        self.readout = nn.Linear(hidden_dims[-1] * len(hidden_dims), output_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        hidden_states = []
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index))
            hidden_states.append(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.readout(torch.cat(hidden_states, dim=1))
        return F.softmax(x, dim=1)


class GCNJumpingKnowledge(nn.Module):
    def __init__(self, input_dim,
                 hidden_dims: list[int],
                 output_dim,
                 dropout=0.5,
                 n_fc_layers: int = 0):
        super(GCNJumpingKnowledge, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.dropout_rate = dropout
        self.n_fc_layers = n_fc_layers

        # First GAT layer
        self.gat_layers.append(GCNConv(input_dim, hidden_dims[0]))

        # Intermediate GAT and FC layers
        for i in range(1, len(hidden_dims)):
            for _ in range(n_fc_layers):
                self.fc_layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i - 1]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
            self.gat_layers.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))

        # Fully connected layers after the last GAT layer
        for _ in range(n_fc_layers):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ))

        # Output layer
        self.readout = nn.Linear(hidden_dims[-1] * len(hidden_dims), output_dim)

    def forward(self, x, edge_index):
        fc_layer_index = 0
        hidden_states = []

        for i, gat_layer in enumerate(self.gat_layers):
            x = torch.relu(gat_layer(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Apply fully connected layers between GAT layers
            for _ in range(self.n_fc_layers):
                x = self.fc_layers[fc_layer_index](x)
                fc_layer_index += 1

            hidden_states.append(x)

        x = self.readout(torch.cat(hidden_states, dim=1))
        return F.log_softmax(x, dim=1)


def init_norm_layer(norm: str):
    if norm == "batch":
        return nn.BatchNorm1d
    elif norm == "layer":
        return nn.LayerNorm
    else:
        return nn.Identity

class GAT_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, norm:str=None):
        super(GAT_Layer, self).__init__()
        self.conv_layer = GATv2Conv(input_dim, output_dim)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        x = self.conv_layer(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


class GATv2(nn.Module):
    def __init__(self, input_dim,
                 hidden_dims: list[int],
                 output_dim,
                 dropout=0.5,
                 n_fc_layers: int = 0,
                 norm: str = None):
        super(GATv2, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.dropout_rate = dropout
        self.n_fc_layers = n_fc_layers
        self.norm = norm

        # First GAT layer
        self.gat_layers.append(GAT_Layer(input_dim, hidden_dims[0], dropout, norm))

        # Intermediate GAT and FC layers
        for i in range(1, len(hidden_dims)):
            # Intermediate FC layers
            for _ in range(n_fc_layers):
                self.fc_layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i - 1]),
                    init_norm_layer(self.norm)(hidden_dims[i - 1]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
            # Intermediate GAT layer
            self.gat_layers.append(GAT_Layer(hidden_dims[i - 1], hidden_dims[i], dropout, norm))

        # Fully connected layers after the last GAT layer
        for _ in range(n_fc_layers):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                init_norm_layer(self.norm)(hidden_dims[i - 1]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, edge_index):
        fc_layer_index = 0

        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)

            # Apply fully connected layers between GAT layers
            for _ in range(self.n_fc_layers):
                x = self.fc_layers[fc_layer_index](x)
                fc_layer_index += 1

        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)
