import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from typing import Union

from src.models.model_constants import MESSAGE_PASSING_MAPPING
from src.models.model_utils import init_norm_layer

def generate_helper_node_type(edge_type):
    return f'{edge_type[0]}->{edge_type[2]}'

class HeteroMessagePassingLayer(nn.Module):

    """
    This class implements a message passing layer for heterogeneous graphs, which can handle different message passing
    types for different edge types. The class is a wrapper around the HeteroConv class from PyTorch Geometric, which
    allows to define different message passing layers for different edge types. The class also supports different
    normalization layers and dropout rates for the message passing layers.
    """

    def __init__(self, output_dim, edge_types: dict[tuple, str], dropout=0.5, norm: str = None):
        """
        Initialize the message passing layer.

        For each edge type, a message passing layer is initialized according to the message passing type specified in
        the edge_types dictionary. The message passing layers are stored in a dictionary, which is used to initialize
        the HeteroConv class from PyTorch Geometric. Depending on the geo torch classes different attributes are
        required for the message passing layers. The class also supports different normalization layers and dropout
        rates for the message passing layers. For RGCN lazy init is not available and the output dimension is assumed to
        be the input dimension.

        :param output_dim: Output dimension of the message passing layer
        :param edge_types: Dictionary of edge types as tuple of strings like ["node_type1", "to", "node_type2"] and
        message passing types as strings
        :param dropout: Dropout rate for the message passing layers
        :param norm: Which normalisation to use. Options are "batch", "layer" or None
        """

        super(HeteroMessagePassingLayer, self).__init__()
        self.edge_types = edge_types

        edge_types_to_remove = []

        hetero_conv_dict = {}
        for i in range(len(edge_types)):
            edge_type = list(edge_types.keys())[i]
            msg_passing_type = list(edge_types.values())[i]

            # Only add self loops for message passing between equal node types
            add_self_loops = edge_type[0] == edge_type[2]

            # Lazy intialization of the input dimension, depending on the message passing class
            input_dim = (-1, -1) if msg_passing_type in ["gat_v2", 'sage'] else -1

            # Collect parameters for message passing layer
            params = {}

            # Init parameters for different message passing types
            if msg_passing_type == "gine" or msg_passing_type == "gin":
                params.update({
                    "nn": nn.Sequential(
                        nn.Linear(output_dim, output_dim),  # TODO: Lazy
                        nn.ReLU()
                    ),
                    "train_eps": True
                })

            if msg_passing_type == "gine" or msg_passing_type == "gat_v2":
                params.update({
                    "edge_dim": 1
                })

            if msg_passing_type == "gcn" or msg_passing_type == "gat_v2":
                params.update({
                    "in_channels": input_dim,
                    "out_channels": output_dim,
                    "add_self_loops": add_self_loops
                })

            if msg_passing_type == "gcn" and edge_type[0] != edge_type[2]:
                # Do not normalise, this causes 0 passes for heterogene passes
                params.update({
                    "normalize": False
                })

            if msg_passing_type == "sage" or msg_passing_type == "cfconv":
                params.update({
                    "in_channels": input_dim,
                    "out_channels": output_dim
                })

            if params == {}:
                raise ValueError(f"Message passing type {msg_passing_type} not supported.")

            # Add edge type with helper node types for gcn and cfconv
            if msg_passing_type in ["gcn", 'cfconv'] and (edge_type[0] != edge_type[2]):
                #edge_types_to_remove.append(edge_type)
                helper_node_type = generate_helper_node_type(edge_type)
                new_edge_type = (helper_node_type, 'to', helper_node_type)

                # Replace new edge type in edge_types
                self.edge_types = {new_edge_type if k == edge_type else k:v for k,v in self.edge_types.items()}
                edge_type = new_edge_type

            # Initialize message passing layer
            message_passing_class = MESSAGE_PASSING_MAPPING[msg_passing_type]
            hetero_conv_dict[edge_type] = message_passing_class(**params)

        # Remove replaced edge types
        for et in edge_types_to_remove:
            del self.edge_types[et]

        self.message_passing_layer = HeteroConv(hetero_conv_dict, aggr="sum")

        if norm == "batch":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()
        self.dropout_rate = dropout

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_attr_dict=None):
        """
        Forward pass of the message passing layer.

        The input dictionary of feature tensors, edge index tensors and edge attribute tensors is used to apply the
        message passing layer to the input features. The output features are then normalized and passed through a ReLU
        activation function. The dropout rate is applied to the output features.

        :param x_dict: Dictionary of input feature tensors
        :param edge_index_dict: Dictionary of edge index tensors
        :param edge_attr_dict: Dictionary of edge attribute tensors (optional)
        :return: Dictionary of output feature tensors
        """


        input_msg_passing = {'x_dict': x_dict, 'edge_index_dict': edge_index_dict.copy(), 'edge_attr_dict': {},
                             'edge_weight_dict': {}}
        for edge_type, msg_passing_type in self.edge_types.items():

            # Merge feature vectors to enable hetero msg passing (for gcn and cfconv)
            if msg_passing_type in ["gcn", 'cfconv'] and ('->' in edge_type[0]):
                # Generate new edge type for helper node
                helper_node_type = edge_type[0]

                edge_source = edge_type[0].split("->")[0]
                edge_target = edge_type[0].split("->")[1]

                old_edge_type = (edge_source, 'to', edge_target)

                # Concat feature vectors into one tensor for gcn on hetero edges
                input_msg_passing['x_dict'].update({helper_node_type:
                                                        torch.cat([x_dict[edge_source],
                                                                   x_dict[edge_target]])
                                                    })

                # Adjust edge index for concatenated feature vectors
                previous_edge_name = (edge_source, 'to', edge_target)
                edge_source_indices = edge_index_dict[old_edge_type][0].unsqueeze(0)
                edge_target_indices = edge_index_dict[old_edge_type][1].unsqueeze(0)
                input_msg_passing['edge_index_dict'].update(
                    {edge_type:
                         torch.cat([edge_source_indices,
                                    edge_target_indices + x_dict[previous_edge_name[0]].shape[0]], dim=0)
                     })

                # Remove old edge type
                input_msg_passing['edge_index_dict'].pop(old_edge_type)

                # Copy edge attribute for new edge type
                if edge_attr_dict is not None:
                    edge_attr_dict[edge_type] = edge_attr_dict[old_edge_type]

            # prepare message passing input, if edges exist for that edge type
            if input_msg_passing['edge_index_dict'][edge_type].shape[1] != 0:

                if msg_passing_type in ["gat_v2","gine"] :
                    input_msg_passing['edge_attr_dict'].update({edge_type: edge_attr_dict[edge_type]})

                if msg_passing_type in ["gcn", 'cfconv']:
                    # Divide by max to invert distance as weights -> higher distance ==> lower weight
                    edge_weight_dict = {edge_type: edge_attr_dict[edge_type].max() - edge_attr_dict[edge_type]}
                    input_msg_passing['edge_weight_dict'].update(edge_weight_dict)

                if msg_passing_type == "rgcn":
                    # Insert edge type tensor (required for rgcn class)
                    input_msg_passing['edge_type'].update({edge_type: torch.zeros(edge_attr_dict[edge_type].shape[0])})


        x_dict = self.message_passing_layer(**input_msg_passing)

        # Aggregate helper node types with regular node types
        for edge_type in self.edge_types.keys():
            if "->" in edge_type[0]:
                edge_target = edge_type[0].split("->")[1]
                n_target_indices = x_dict[edge_target].size()[0]
                x_dict[edge_target] = x_dict[edge_target] + x_dict[edge_type[0]][-n_target_indices:]
                x_dict.pop(edge_type[0])

        x_dict = {key: self.norm(x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Dropout only applied in fc between message passings
        # x_dict = {key: F.dropout(x, p=self.dropout_rate, training=self.training) for key, x in x_dict.items()}

        return x_dict


class HeteroGNN(nn.Module):
    def __init__(self,
                 output_dim: int,
                 cell_types: list[str],
                 msg_passing_types: Union[dict[str, str],str],
                 hidden_dims: list[int] = None,
                 hidden_dim: int = None,
                 n_message_passings: int = None,
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
                edge_types=edge_types,
                dropout=dropout,
                norm=norm,
            ))

            # Intermediate FC layers
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
        self.output_layer = nn.LazyLinear(output_dim)

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


class HeteroGnnJK(HeteroGNN):

    def __init__(self, **kwargs):

        super(HeteroGnnJK ,self).__init__(**kwargs)

        self.last_layer = nn.Sequential(
            nn.LazyLinear(self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(p=kwargs['dropout']),
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(p=kwargs['dropout']),
            nn.Linear(self.hidden_dims[-1], kwargs['output_dim'])
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        fc_layer_index = 0

        # Apply one FC to unify number of featues for all node types
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
        fc_layer_index += 1

        # Get glom feature for jumping knowledge
        x_glom = x_dict['glomeruli']

        for i, message_passing_layer in enumerate(self.message_passing_layers):
            x_dict = message_passing_layer(x_dict, edge_index_dict, edge_attr_dict)

            # Apply fully connected layers between GAT layers
            for _ in range(self.n_fc_layers):
                for node_type, x in x_dict.items():
                    x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
                fc_layer_index += 1

        x = self.output_layer(torch.cat((x_dict['glomeruli'], x_glom), dim=1))

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

