from torch import nn
from torchvision import models

from src.models.model_utils import init_norm_layer


def initialize_resnet(output_dim: int,
                      hidden_dims:list[int],
                      image_size: int,
                      device,
                      pre_trained: bool = True,
                      dropout: float = 0.5,
                      fc_norm: str = None):
    """
    Initialize a ResNet model.

    Args:
        pre_trained (bool): If True, the model will be initialized with pre-trained weights.

    Returns:
        models.resnet.ResNet: The initialized ResNet model.
    """
    model = models.resnet18(pretrained=pre_trained)

    latent_dim = model.fc.in_features
    fc_modules = nn.ModuleList()

    test = model.fc

    # First layer
    fc_modules.append(nn.Sequential(
        nn.Linear(latent_dim, hidden_dims[0]),
        init_norm_layer(fc_norm)(hidden_dims[0]),
        nn.ReLU(),
        nn.Dropout(p=dropout)
    ))

    # Hidden layer
    for i in range(1, len(hidden_dims)):
        fc_modules.append(nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    init_norm_layer(fc_norm)(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))

    # Last layer
    fc_modules.append(nn.Linear(hidden_dims[-1], output_dim))

    model.fc = nn.Sequential(*fc_modules)

    return model
