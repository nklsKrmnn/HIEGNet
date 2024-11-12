from torch import nn
from torchvision import models

from models.model_constants import EFFICIENT_NET_MAPPING
from src.models.model_utils import init_norm_layer
from src.models.model_constants import RESNET_MODEL_MAPPING


def initialize_resnet(output_dim: int,
                      hidden_dims: list[int],
                      image_size: int,
                      layers: int,
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
    model = RESNET_MODEL_MAPPING[layers](pretrained=pre_trained)

    latent_dim = model.fc.in_features
    fc_modules = nn.ModuleList()

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


def initialize_efficientnet_v2(output_dim: int,
                               enet_size: str,
                               image_size: int,
                               device,
                               dropout: float = 0.5):
    """
    Initialize a ResNet model.

    Args:
        pre_trained (bool): If True, the model will be initialized with pre-trained weights.

    Returns:
        models.resnet.ResNet: The initialized ResNet model.
    """
    model = EFFICIENT_NET_MAPPING[enet_size](weights='DEFAULT')

    # Modify the classifier to fit the number of output classes
    num_features = model.classifier[1].in_features  # Get the number of input features to the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 32),  # Replace 'num_classes' with the number of your output classes
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, output_dim)
    )

    return model
