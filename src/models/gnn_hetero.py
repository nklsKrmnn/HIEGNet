import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, HeteroConv

from src.models.model_constants import MESSAGE_PASSING_MAPPING
from src.models.gnn_models import MessagePassingLayer
from src.models.model_utils import init_norm_layer


class HeteroMessagePassingLayer(nn.Module):
    def __init__(self, output_dim, edge_types, dropout=0.5, norm: str = None):
        super(HeteroMessagePassingLayer, self).__init__()
        self.edge_types = edge_types

        hetero_conv_dict = {}
        for edge_type, msg_passing_type in edge_types.items():
            # Only add self loops for message passing between equal node types
            add_self_loops = edge_type[0] == edge_type[2]

            # Lazy intialization of the input dimension, depending on the message passing class
            input_dim = (-1, -1) if msg_passing_type == "gat_v2" else -1

            # Collect parameters for message passing layer
            if msg_passing_type == "gin":
                params = {
                    "nn": nn.LazyLinear(output_dim),
                    "train_eps": True
                }

            elif msg_passing_type == "gcn":
                params = {
                    "in_channels": input_dim,
                    "out_channels": output_dim,
                    "add_self_loops": add_self_loops
                }
            elif msg_passing_type == "gat_v2":
                params = {
                    "in_channels": input_dim,
                    "out_channels": output_dim,
                    "add_self_loops": add_self_loops,
                    "edge_dim": 1
                }
            else:
                raise ValueError(f"Message passing type {msg_passing_type} not supported.")

            # Initialize message passing layer
            message_passing_class = MESSAGE_PASSING_MAPPING[msg_passing_type]
            hetero_conv_dict[edge_type] = message_passing_class(**params)

        self.message_passing_layer = HeteroConv(hetero_conv_dict, aggr="sum")

        if norm == "batch":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()
        self.dropout_rate = dropout

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        input_msg_passing = {'x_dict': x_dict, 'edge_index_dict': edge_index_dict, 'edge_attr_dict': {},
                             'edge_weight_dict': {}}
        for edge_type, msg_passing_type in self.edge_types.items():
            if msg_passing_type == "gat_v2":
                input_msg_passing['edge_attr_dict'].update({edge_type: edge_attr_dict[edge_type]})
            if msg_passing_type == "gcn":
                input_msg_passing['edge_weight_dict'].update({edge_type: edge_attr_dict[edge_type]})

        x_dict = self.message_passing_layer(**input_msg_passing)
        x_dict = {key: self.norm(x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        # x_dict = {key: F.dropout(x, p=self.dropout_rate, training=self.training) for key, x in x_dict.items()}

        return x_dict


class HeteroGNN(nn.Module):
    def __init__(self,
                 output_dim: int,
                 cell_types: list[str],
                 msg_passing_types: dict[str, str],
                 hidden_dims: list[int] = None,
                 hidden_dim: int = None,
                 n_message_passings=None,
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
        hidden_dims = [hidden_dim for _ in range(n_message_passings)] if hidden_dims is None else hidden_dims

        # Determine existing node and edge types
        edge_types = {('glomeruli', 'to', 'glomeruli'): msg_passing_types['glom_to_glom']}
        for cell_type in cell_types:
            edge_types[(cell_type, 'to', 'glomeruli')] = msg_passing_types['cell_to_glom']
            for cell_type2 in cell_types:
                edge_types[(cell_type, 'to', cell_type2)] = msg_passing_types['cell_to_cell']
        node_types = ['glomeruli'] + cell_types

        # FC layer to unify input dimensions
        lin_dict = nn.ModuleDict()
        for node_type in node_types:
            lin_dict[node_type] = nn.Sequential(
                nn.LazyLinear(hidden_dims[0]),
                init_norm_layer(self.norm_fc_layers)(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        self.fc_layers.append(lin_dict)

        # GAT and FC layers
        for i in range(0, len(hidden_dims)):
            # Intermediate message passing layer
            self.message_passing_layers.append(HeteroMessagePassingLayer(
                output_dim=hidden_dims[i],
                edge_types=edge_types,
                dropout=dropout,
                norm=norm,
            ))

            # Intermediate FC layers
            for _ in range(n_fc_layers):
                lin_dict = nn.ModuleDict()
                for node_type in node_types:
                    lin_dict[node_type] = nn.Sequential(
                        nn.Linear(hidden_dims[i - 1], hidden_dims[i - 1]),
                        init_norm_layer(self.norm_fc_layers)(hidden_dims[i - 1]),
                        nn.ReLU(),
                        nn.Dropout(p=dropout)
                    )
                self.fc_layers.append(lin_dict)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

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

        output = self.output_layer(x_dict['glomeruli'])
        if self.softmax_function == "softmax":
            output = F.softmax(output, dim=1)
        elif self.softmax_function == "log_softmax":
            output = F.log_softmax(output, dim=1)
        return output
