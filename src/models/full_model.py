import torch
from torch import nn as nn
from torch.nn import functional as F

from src.models.cnn import CNN
from src.models.gnn_hetero import HeteroGNN


class FullHybrid(HeteroGNN):
    def __init__(self,
                 output_dim: int,
                 cell_types: list[str],
                 msg_passing_types: dict[str, str],
                 cnn_params: dict,
                 image_size: int,
                 hidden_dims: list[int] = None,
                 hidden_dim: int = None,
                 n_message_passings: int = None,
                 dropout=0.5,
                 n_fc_layers: int = 0,
                 norm: str = None,
                 norm_fc_layers: str = None,
                 device=torch.device('cpu'),
                 softmax_function: "str" = "softmax"):

        super(FullHybrid, self).__init__(
            output_dim=output_dim,
            cell_types=cell_types,
            msg_passing_types=msg_passing_types,
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            n_message_passings=n_message_passings,
            dropout=dropout,
            n_fc_layers=n_fc_layers,
            norm=norm,
            norm_fc_layers=norm_fc_layers,
            softmax_function=softmax_function
        )

        # Init CNN
        cnn_params['cnn_softmax_function'] = 'none'
        cnn_params['image_size'] = image_size
        cnn_params = cnn_params.copy()
        self.cnn_encoder = CNN(**cnn_params)



    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # CNN
        image = x_dict['glomeruli_image']
        cnn_output = self.cnn_encoder(image)
        cnn_output = torch.flatten(cnn_output, start_dim=1)
        cnn_output = F.relu(cnn_output)

        # Delete image features
        del x_dict['glomeruli_image']

        # Concat CNN output to glomeruli node features
        x_dict['glomeruli'] = torch.cat([x_dict['glomeruli'], cnn_output], dim=1)

        fc_layer_index = 0

        # Apply one FC to unify number of featues for all node types
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
        fc_layer_index += 1

        # Apply message passing layers
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

class FullHybridJK(FullHybrid):
    def __init__(self,
                 output_dim: int,
                 cell_types: list[str],
                 msg_passing_types: dict[str, str],
                 cnn_params: dict,
                 image_size: int,
                 hidden_dims: list[int] = None,
                 hidden_dim: int = None,
                 n_message_passings: int = None,
                 dropout=0.5,
                 n_fc_layers: int = 0,
                 norm: str = None,
                 norm_fc_layers: str = None,
                 device=torch.device('cpu'),
                 softmax_function: "str" = "softmax"):

        super(FullHybridJK, self).__init__(
            output_dim=output_dim,
            cell_types=cell_types,
            msg_passing_types=msg_passing_types,
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            n_message_passings=n_message_passings,
            dropout=dropout,
            n_fc_layers=n_fc_layers,
            norm=norm,
            norm_fc_layers=norm_fc_layers,
            softmax_function=softmax_function,
            cnn_params=cnn_params,
            image_size=image_size
        )

        self.output_layer = nn.Sequential(
            nn.LazyLinear(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim)
        )



    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # CNN
        image = x_dict['glomeruli_image']
        cnn_output = self.cnn_encoder(image)
        cnn_output = torch.flatten(cnn_output, start_dim=1)
        cnn_output = F.relu(cnn_output)

        # Delete image features
        del x_dict['glomeruli_image']

        # Concat CNN output to glomeruli node features
        x_dict['glomeruli'] = torch.cat([x_dict['glomeruli'], cnn_output], dim=1)

        fc_layer_index = 0

        # Apply one FC to unify number of featues for all node types
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
        fc_layer_index += 1

        # Apply message passing layers
        for i, message_passing_layer in enumerate(self.message_passing_layers):
            x_dict = message_passing_layer(x_dict, edge_index_dict, edge_attr_dict)

            # Apply fully connected layers between GAT layers
            for _ in range(self.n_fc_layers):
                for node_type, x in x_dict.items():
                    x_dict[node_type] = self.fc_layers[fc_layer_index][node_type](x)
                fc_layer_index += 1

        x = self.output_layer(torch.cat([x_dict['glomeruli'], cnn_output],dim=1))

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
