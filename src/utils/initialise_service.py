from torch import optim
from torch import nn
import warnings


def init_optimizer(epochs: int,
                   learning_rate: float,
                   lr_scheduler_params: dict,
                   model: nn.Module,
                   momentum: float,
                   optimizer: str,
                   weight_decay: float,
                   fc_learning_rate: float=None) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Initializes the optimizer and learning rate scheduler for the model.

    Args:
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        lr_scheduler_params (dict): The learning rate scheduler parameters.
        model (nn.Module): The PyTorch model to train.
        momentum (float): The momentum for the optimizer.
        optimizer (str): The optimizer to use.
        weight_decay (float): The weight decay for the optimizer.
        fc_learning_rate (float): The learning rate for the fully connected layer.

    Returns:
        tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]: The optimizer and learning rate scheduler.
    """

    # Setting up the model parameters as input for optimizer
    if fc_learning_rate is not None:
        if "fc" in [p[0] for p in model.named_parameters()]:
            model_params_list = [
                {"params": model.fc.parameters(), "lr": fc_learning_rate},
                {"params": [p[1] for p in model.named_parameters() if 'fc' not in p[0]]}
            ]
        elif 'classifier' in [p[0] for p in model.named_parameters()]:
            model_params_list = [
                {"params": model.classifier.parameters(), "lr": fc_learning_rate},
                {"params": [p[1] for p in model.named_parameters() if 'classifier' not in p[0]]}
            ]
        else:
            warnings.warn(f"[TRAINER]: No fully connected layer found in model, defaulting to standard parameters.")
            model_params_list = model.parameters()
    else:
        model_params_list = model.parameters()

    # Setting up the optimizer
    if optimizer == "adam":
        optimizer_instance = optim.Adam(
            model_params_list, lr=learning_rate, weight_decay=weight_decay)
        if momentum != 0:
            print(f"[TRAINER]: Momentum {momentum} is not used since the optimizer is set to Adam")
    elif optimizer == "sgd":
        optimizer_instance = optim.SGD(
            model_params_list, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    else:
        print(f"[TRAINER]: Optimizer {optimizer} is not valid, defaulting to Adam")
        optimizer_instance = optim.Adam(
            model.parameters(), lr=learning_rate)

    # Setting up the learning rate scheduler
    if lr_scheduler_params is not None:
        if lr_scheduler_params["scheduler"] == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance,
                                                                **lr_scheduler_params["params"])
        elif lr_scheduler_params["scheduler"] == "CyclicLR":
            lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer_instance,
                                                       **lr_scheduler_params["params"])
        elif lr_scheduler_params["scheduler"] == "OneCycleLR":
            total_steps = epochs
            lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer_instance,
                                                         total_steps=total_steps,
                                                         **lr_scheduler_params["params"])
        else:
            print(
                f"[TRAINER]: Learning rate scheduler {lr_scheduler_params['scheduler']} is not valid, no scheduler is used.")
            lr_scheduler = None
    else:
        lr_scheduler = None

    return optimizer_instance, lr_scheduler