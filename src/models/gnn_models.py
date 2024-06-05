import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims:list[int], output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
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
        hidden_dim=hidden_dims[0]
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

class GCNJumpingKnowledge(nn.Module):
    def __init__(self, input_dim, hidden_dims:list[int], output_dim, dropout=0.5):
        super(GCNJumpingKnowledge, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.readout = nn.Linear(hidden_dims[-1]*len(hidden_dims), output_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        hidden_states = []
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index))
            hidden_states.append(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.readout(torch.cat(hidden_states, dim=1))
        return F.softmax(x, dim=1)


class GATv2(nn.Module):
    def __init__(self, input_dim, hidden_dims:list[int], output_dim, dropout=0.5):
        super(GATv2, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATv2Conv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GATv2Conv(hidden_dims[i-1], hidden_dims[i]))
        self.readout = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_rate = dropout

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.readout(x)
        return F.softmax(x, dim=1)
