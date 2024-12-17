import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SchNetMessagePassing(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cutoff=torch.inf) -> None: # cutoff is set to infinity by default and usually not used, as we deal with this in the edge creation
        super(SchNetMessagePassing, self).__init__(aggr="add")  # Use sum aggregation (default for SchNet)

        self.cutoff = cutoff
        if in_channels == -1:
            self.embedding = nn.LazyLinear(out_channels)
        else:
            self.embedding = nn.Linear(in_channels, out_channels)

        # Learnable function of distance for weighting messages
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, out_channels)
        )

    def forward(self, x, edge_index, edge_weight, **kwargs):
        """
        Forward pass for message passing.

        Args:
            x (Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
            edge_weight (Tensor): Edge weights (distances) of shape [num_edges].

        Returns:
            Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        # Initial embedding for node features
        x = self.embedding(x)

        # Perform message passing
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        """
        Compute messages from neighbors.

        Args:
            x_j (Tensor): Neighbor node features of shape [num_edges, out_channels].
            edge_weight (Tensor): Edge weights (distances) of shape [num_edges].

        Returns:
            Tensor: Messages of shape [num_edges, out_channels].
        """
        # Apply cutoff function (optional for distances larger than cutoff)
        cutoff_filter = (edge_weight < self.cutoff).float()

        # Expand edge weight dimension for MLP
        edge_weight = edge_weight.view(-1, 1) * cutoff_filter.view(-1, 1)

        # Compute distance-based weight using MLP
        weight = self.distance_mlp(edge_weight)

        # Weight neighbor features
        return weight * x_j



