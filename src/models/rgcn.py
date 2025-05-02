import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from typing import Union

from src.models.model_constants import MESSAGE_PASSING_MAPPING
from src.models.model_utils import init_norm_layer

class HeteroGNN(nn.Module):
    def __init__(self,
                 output_dim: int,
                 cell_types: list[str],
                 hidden_dims: list[int] = None,
                 hidden_dim: int = None,
                 n_message_passings: int = None,
                 n_readout_layers: int = 1,
                 dropout=0.5,
                 n_fc_layers: int = 0,
                 norm: str = None,
                 norm_fc_layers: str = None,
                 softmax_function: "str" = "softmax"):
        super(HeteroGNN, self).__init__()
        self.message_passing_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.dropout_rate = dropout
        self.n_fc_layers = n_fc_layers
        self.norm = norm
        self.norm_fc_layers = norm_fc_layers
        self.softmax_function = softmax_function

        # Create hidden_dims list from dimension and number message_passing_steps if hidden_dims is not given
        self.hidden_dims = [hidden_dim for _ in range(n_message_passings)] if hidden_dims is None else hidden_dims

        # Determine existing node and edge types
        edge_types = {('glomeruli', 'to', 'glomeruli'): msg_passing_types['glom_to_glom']}
        for cell_type in cell_types:
            edge_types[(cell_type, 'to', 'glomeruli')] = msg_passing_types['cell_to_glom']
            edge_types[('glomeruli', 'to', cell_type)] = msg_passing_types['cell_to_glom']
            for cell_type2 in cell_types:
                edge_types[(cell_type, 'to', cell_type2)] = msg_passing_types['cell_to_cell']
        node_types = ['glomeruli'] + cell_types

        # FC layer to unify input dimensions
        lin_dict = nn.ModuleDict()
        for node_type in node_types:
            lin_dict[node_type] = nn.Sequential(
                nn.LazyLinear(self.hidden_dims[0]),
                init_norm_layer(self.norm_fc_layers)(self.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        self.fc_layers.append(lin_dict)

        # Message passings and FC layers
        for i in range(0, len(self.hidden_dims)):
            # Intermediate message passing layer
            self.message_passing_layers.append(HeteroMessagePassingLayer(
                output_dim=self.hidden_dims[i],
                msgp_type_dict=edge_types,
                dropout=dropout,
                norm=norm,
            ))

            # Update FC layers
            for _ in range(n_fc_layers):
                lin_dict = nn.ModuleDict()
                for node_type in node_types:
                    lin_dict[node_type] = nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i - 1]),
                        init_norm_layer(self.norm_fc_layers)(self.hidden_dims[i - 1]),
                        nn.ReLU(),
                        nn.Dropout(p=dropout)
                    )
                self.fc_layers.append(lin_dict)

        # Output layer
        self.output_layer = []

        for _ in range(n_readout_layers - 1):
            self.output_layer.append(nn.Sequential(
                nn.LazyLinear(self.hidden_dims[-1]),
                init_norm_layer(self.norm_fc_layers)(self.hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ))
        self.output_layer.append(nn.LazyLinear(output_dim))
        self.output_layer = nn.Sequential(*self.output_layer)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        fc_layer_index = 0

        # Apply one FC to unify number of featues for all node types
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
        fc_layer_index += 1

        for i, message_passing_layer in enumerate(self.message_passing_layers):
            x_dict = message_passing_layer(x_dict, edge_index_dict, edge_attr_dict)

            # Apply fully connected layers between GAT layers
            for _ in range(self.n_fc_layers):
                for node_type, x in x_dict.items():
                    x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
                fc_layer_index += 1

        x = self.output_layer(x_dict['glomeruli'])

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