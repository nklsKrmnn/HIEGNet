from torch import nn
import torch.nn.functional as F

from src.models.model_utils import init_norm_layer

class MLP(nn.Module):

    def __init__(self,
                 output_dim,
                 hidden_dims: list[int]=None,
                 mlp_dropout=0.5,
                 input_dim: int = None,
                 mlp_hidden_dim: int = None,
                 mlp_hidden_layers: int = None,
                 mlp_softmax_function: str = "softmax",
                 norm: str = None):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout_rate = mlp_dropout
        self.norm = norm
        self.softmax_function = mlp_softmax_function

        if hidden_dims is None:
            hidden_dims = [mlp_hidden_dim for _ in range(mlp_hidden_layers)]

        if input_dim is not None:
            hidden_dims = [input_dim] + hidden_dims
        hidden_dims = hidden_dims + [output_dim]

        # Lazy first layer if required
        if input_dim is None:
            self.fc_layers.append(nn.Sequential(
                nn.LazyLinear(hidden_dims[0]),
                init_norm_layer(self.norm)(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout)
            ))

        # FC layers
        for i in range(1, len(hidden_dims)):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                init_norm_layer(self.norm)(hidden_dims[i]),
                nn.LeakyReLU(),
                nn.Dropout(p=mlp_dropout)
            ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, *args):

        for i, layer in enumerate(self.fc_layers):
            x = layer(x)

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

class MLP_dumm(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes the MLP model.

        Args:
            input_size (int): The size of the input features.
            output_size (int): The size of the output features.
        """
        super(MLP_dumm, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        return output
