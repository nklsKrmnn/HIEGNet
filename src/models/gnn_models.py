import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the GCN model
class TestGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(TestGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        #TODO put a dense layer after (read out: FC layer notewise)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)