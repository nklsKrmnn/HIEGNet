"""
This module contains the Trainer class which is used to train a PyTorch model.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.evaluation.test_scores import calc_test_scores
from src.utils.model_service import ModelService
from src.utils.logger import Logger

FIG_OUTPUT_PATH: Final[Path] = Path("./data/output/eval_plot")

# create directory if it does not exist
FIG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class Trainer:
    """
    A class used to represent a Trainer for a PyTorch Geometric model.

    This class handles the training process for a PyTorch Geometric model, initialising the loss and the optimizer
    moving the model and loss function to the GPU if available, setting up the data loaders,
    and training the model for a specified number of epochs.

    Attributes:
        batch_size (int): The batch size for training.
        test_split (float): The fraction of the data to use for test.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        lr_scheduler (optim.lr_scheduler.ReduceLROnPlateau | optim.lr_scheduler.CyclicLR | optim.lr_scheduler.OneCycleLR): The learning rate scheduler to use.
        loss (nn.MSELoss | nn.CrossEntropyLoss | nn.NLLLoss): The loss function to use.
        weight_decay (float): The weight decay for the optimizer.
        balance_classes (bool): Whether to balance the classes.
        optimizer (optim.SGD | optim.Adam): The optimizer to use.
        device (torch.device): Whether to use the CPU or the GPU.
        model (nn.Module): The PyTorch model to train.
        logger (Logger): The logger to use for logging training information.
        eval_mode (bool): Is set to True, if the evaluation function is called.
        seed (int): The seed for the random number generator.
        batch_shuffle (bool): Whether to shuffle the batches.
        patience (int): The number of epochs to wait for improvement before stopping.
        log_image_frequency (int): The frequency of logging images.
        dataset (Dataset): The dataset to use for training and validation.
        test_split (float): The fraction of the data to use for testing.
    """
    batch_size: int
    test_split: float
    epochs: int
    learning_rate: float
    loss: Union[nn.MSELoss, nn.CrossEntropyLoss, nn.NLLLoss]
    optimizer: Union[optim.SGD, optim.Adam]
    lr_scheduler: Union[
        optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR, None]
    weight_decay: float
    balance_classes: bool
    device: torch.device
    model: nn.Module
    logger: Logger
    eval_mode: bool
    seed: int
    batch_shuffle: bool
    patience: int
    log_image_frequency: int
    dataset: Dataset
    test_split: float

    def __init__(
            self,
            dataset,
            model: nn.Module,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            device: torch.device,
            logger: Logger,
            test_split: float = 0.2,
            loss: str = "mse",
            optimizer: str = "adam",
            lr_scheduler_params: dict = None,
            weight_decay: float = 0,
            balance_classes: bool = False,
            momentum: float = 0,
            eval_mode: bool = False,
            seed: int = None,
            batch_shuffle: bool = False,
            patience: int = 50,
            log_image_frequency: int = 10
    ):
        """
        Creates a Trainer instance from an unpacked configuration file.
        This method sets up the loss function, optimizer and the lr scheduler based on the provided
        parameters. The other parameters from the config are simply passed through to the Trainer instance.

        Args:
            dataset: Dataset to be used for training, optimizing and validating the model.
            model (nn.Module): Model to be trained.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for the optimizer.
            test_split (float): Fraction of the data to use for testing.
            device (torch.device): Whether to use the CPU or the GPU for training.
            logger (Logger): Logger to use for logging training information.
            loss (str, optional): Loss function to use. Defaults to "mse".
            optimizer (str, optional): Optimizer to use. Defaults to "adam".
            lr_scheduler_params (dict, optional): Parameters for the learning rate scheduler. Defaults to None.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.
            balance_classes (bool, optional): Whether to balance the classes. Defaults to False.
            momentum (float, optional): Momentum for the optimizer. Defaults to 0.
            eval_mode (bool, optional): If the model is evaluated, the validation split is set to 1.
            seed (int, optional): Seed for the random number generator.
            batch_shuffle (bool): Whether to shuffle the batches.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 50.
            log_image_frequency (int, optional): Frequency of logging images. Defaults to 10.

        Returns:
            Trainer: A Trainer instance with the specified configuration.
        """

        # Compute class weights
        if balance_classes:
            class_weights_tensor = dataset.get_class_weights()
        else:
            class_weights_tensor = None

        # Setting up the loss
        if loss == "mse":
            loss_instance = nn.MSELoss()
        elif loss == "crossentropy":
            loss_instance = nn.CrossEntropyLoss(weight=class_weights_tensor)
        elif loss == "nll":
            loss_instance = nn.NLLLoss(weight=class_weights_tensor)
        else:
            print(f"[TRAINER]: Loss {loss} is not valid, defaulting to MSELoss")
            loss_instance = nn.MSELoss()
        print(f"[TRAINER]: Using {loss} loss function.")

        # Setting up the optimizer
        if optimizer == "adam":
            optimizer_instance = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if momentum != 0:
                print(f"[TRAINER]: Momentum {momentum} is not used since the optimizer is set to Adam")
        elif optimizer == "sgd":
            optimizer_instance = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
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
                total_steps = (int(math.ceil(len(dataset) / batch_size))) * epochs
                lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer_instance,
                                                             total_steps=total_steps,
                                                             **lr_scheduler_params["params"])
            else:
                print(
                    f"[TRAINER]: Learning rate scheduler {lr_scheduler_params['scheduler']} is not valid, no scheduler is used.")
                lr_scheduler = None
        else:
            lr_scheduler = None

        # Setting random seed for torch
        if seed is not None:
            torch.manual_seed(seed)
            if device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss_instance
        self.optimizer = optimizer_instance
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.device = device
        self.model = model
        self.logger = logger
        self.eval_mode = eval_mode
        self.seed = seed
        self.batch_shuffle = batch_shuffle
        self.patience = patience
        self.log_image_frequency = log_image_frequency
        self.logger = logger
        self.dataset = dataset
        self.test_split = test_split
        print("[TRAINER]: Trainer was successfully set up.")

    def start_training(self) -> None:
        """
        This is the entrypoint method to start the training process for the model.

        This method first moves the model and loss function to the device. The
        method sets up the data loaders using the `setup_dataloaders`
        method. Afterward, it starts the actual training using the `train_model` method and
        logs the reason for finishing the training. After the training process is finished,
        the method closes the logger.
        """
        self.model.to(self.device)
        self.loss.to(self.device)

        # Creating training and validation data loaders from the given data
        # source
        train_loader, validation_loader, test_loader = self.setup_dataloaders()

        # Perform model training
        self.logger.write_training_start()
        finish_reason = self.train_model(train_loader, validation_loader, test_loader)
        self.logger.write_training_end(finish_reason)

    def setup_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Sets up the data loaders holding the graph dataset.

        Returns:
            DataLoader: The training and test data loader.
        """
        # get indices of train and test patients
        train_indices, validation_indices, test_indices = self.dataset.get_set_indices()
        if validation_indices == []:
            validation_indices = train_indices
        train_dataset = Subset(self.dataset, train_indices)
        validation_dataset = Subset(self.dataset, validation_indices)
        test_dateset = Subset(self.dataset, test_indices)

        # Create torch data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dateset, batch_size=self.batch_size) if test_indices != [] else None

        return train_loader, validation_loader, test_loader

    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader) -> str:
        """
        Trains the model for a specified number of epochs. For each epoch, the method calculates
        the training loss and validation loss, logs these losses, and saves the current state
        of the model.

        If a `KeyboardInterrupt` is raised during training, the method catches it and sets the
        finish reason to `"Training interrupted by user"`. If the training finishes without
        interruption, the finish reason is set to `"Training finished normally"`.

        Args:
            train_loader (DataLoader): DataLoader for the training graphs.
            validation_loader (DataLoader): DataLoader for the validation graphs.
            test_loader (DataLoader): DataLoader for the test graphs.

        Returns:
            str: The reason the training ended.
        """
        # Setup for early stopping
        min_loss = float('inf')
        cur_patience = 0

        finish_reason = "Training terminated before training loop ran through."
        for epoch in tqdm(range(self.epochs)):
            try:
                train_loss, train_results = self.train_step(train_loader)

                # Logging loss of results
                self.logger.log_loss(train_loss, epoch, "1_train")

                # Calculating validation loss
                val_loss, val_results = self.validation_step(validation_loader)
                self.logger.log_loss(val_loss, epoch, "2_validation")

                if test_loader != None:
                    test_scores, test_results = self.test_step(test_loader, "test")
                else:
                    test_results = (val_results[0], val_results[1])
                    test_scores = calc_test_scores(val_results[1], val_results[0])
                for score, score_dict in test_scores.items():
                    for class_label, value in score_dict.items():
                        self.logger.log_test_score(value, epoch, class_label, score)

                if epoch % self.log_image_frequency == 0:
                    self.visualize(train_results[0], train_results[1], 'train', epoch)
                    self.visualize(val_results[0], val_results[1], "val", epoch)
                    self.visualize(test_results[0], test_results[1], 'test', epoch)

                # Logging learning rate (getter-function only works with torch2.2 or higher)
                if self.lr_scheduler is not None:
                    try:
                        self.logger.log_lr(lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    except AttributeError:
                        self.logger.log_lr(lr=self.lr_scheduler.optimizer.param_groups[0]['lr'], epoch=epoch)

                # Early stopping
                if min_loss > val_loss:
                    min_loss = val_loss
                    cur_patience = 0
                else:
                    if self.patience > 0:
                        cur_patience += 1
                        if cur_patience == self.patience:
                            finish_reason = "Training finished because of early stopping."
                            self.save_model()
                            break
                self.save_model()

            except KeyboardInterrupt:
                finish_reason = "Training interrupted by user input."
                break

        # Overwrite finish reason if training was not finished due to early
        # stopping or user input
        if finish_reason == "Training terminated before training loop ran through.":
            finish_reason = "Training was normally completed."

        return finish_reason

    def train_step(self, dataloader) -> tuple[float, float]:
        """
        Calculates the training loss for the model. This method is called during each epoch.

        This method iterates over each batch in the training loader. For each batch, it resets the optimizer,
        calculates the loss between the predictions and the actual targets, performs backpropagation, and updates the
        model's parameters. The forward function is computed for the whole dataset. The train and test mask are used
        to separate the dataset into training and test data. The method accumulates the total training loss and
        returns the average training loss per batch.

        Returns:
            float: The average training loss per batch.
            float: The average test loss per batch.
        """
        self.model.train()
        total_train_loss: float = 0
        step_count: int = 0
        complete_predictions = []
        complete_targets = []

        for graph_data in dataloader:

            # Reset optimizer
            self.optimizer.zero_grad()

            # Move data to device
            if isinstance(graph_data.x, torch.Tensor):
                input_graph_feature = graph_data.x.to(self.device)
            else:
                input_graph_feature = graph_data.x
            input_graph_edge_index = graph_data.edge_index.to(self.device)
            target = graph_data.y.to(self.device)

            # Get predictions and train loss
            prediction = self.model.forward(input_graph_feature, input_graph_edge_index)
            train_loss = self.loss(prediction[graph_data.train_mask], target[graph_data.train_mask])

            # Backpropagation
            train_loss.backward()
            self.optimizer.step()

            total_train_loss += train_loss.item()
            step_count += 1

            # Step for ReduceLROnPlateau schedule is done with validation loss
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler,
                                                                optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

            if len(graph_data.y.shape) == 1:
                pred = prediction.detach().argmax(dim=1).cpu()
                targ = target.detach().cpu()
            elif graph_data.y.shape[1] > 1:
                pred = prediction.detach().argmax(dim=1).cpu()
                targ = target.detach().argmax(dim=1).cpu()

            complete_predictions.append(pred[graph_data.train_mask])
            complete_targets.append(targ[graph_data.train_mask])

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_train_loss = total_train_loss / step_count


        return total_train_loss, (complete_predictions, complete_targets)

    def validation_step(self, validation_loader) -> float:
        """
        Calculates the target metric for the test set and generates visualisations for the train and the test set. This method is called in the frequency given in the config.

        This method iterates over each batch in the dataloader, computes the model's
        predictions for the batch, calculates the accuracy between the predictions and the actual
        targets, and accumulates the total accuracy. The method returns the average
        accuracy over all batches.

        Returns:
            float: The loss over all batches.
            float: The accuracy over all batches.
            float: Validation loss
        """
        self.model.eval()
        total_val_loss: float = 0
        step_count: int = 0
        complete_predictions = []
        complete_targets = []

        with torch.no_grad():
            for graph_data in validation_loader:

                if isinstance(graph_data.x, torch.Tensor):
                    input_graph_feature = graph_data.x.to(self.device)
                else:
                    input_graph_feature = graph_data.x
                input_graph_edge_index = graph_data.edge_index.to(self.device)
                target = graph_data.y.float().to(self.device)

                prediction = self.model.forward(input_graph_feature, input_graph_edge_index)
                val_loss = self.loss(prediction[graph_data.val_mask], target[graph_data.val_mask])

                total_val_loss += val_loss.item()
                step_count += 1

                if len(graph_data.y.shape) == 1:
                    pred = prediction.detach().argmax(dim=1).cpu()
                    targ = target.detach().cpu()
                elif graph_data.y.shape[1] > 1:
                    pred = prediction.detach().argmax(dim=1).cpu()
                    targ = target.detach().argmax(dim=1).cpu()

                complete_predictions.append(pred[graph_data.val_mask])
                complete_targets.append(targ[graph_data.val_mask])

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_val_loss = total_val_loss / step_count

        return total_val_loss, (complete_predictions, complete_targets)

    def test_step(self, test_loader, mask_str: str = "test") -> dict[str, dict[str, float]]:
        """
        Calculates the target metric for the test set and generates visualisations for the train and the test set. This method is called in the frequency given in the config.

        This method iterates over each batch in the dataloader, computes the model's
        predictions for the batch, calculates the accuracy between the predictions and the actual
        targets, and accumulates the total accuracy. The method returns the average
        accuracy over all batches.

        Returns:
            float: The loss over all batches.
            float: The accuracy over all batches.
            dict[str, float]: The partial accuracies for each class.
        """
        self.model.eval()
        complete_predictions = []
        complete_targets = []

        with torch.no_grad():
            for graph_data in test_loader:

                if isinstance(graph_data.x, torch.Tensor):
                    input_graph_feature = graph_data.x.to(self.device)
                else:
                    input_graph_feature = graph_data.x
                input_graph_edge_index = graph_data.edge_index.to(self.device)
                target = graph_data.y.float().to(self.device)

                prediction = self.model.forward(input_graph_feature, input_graph_edge_index)

                if len(graph_data.y.shape) == 1:
                    pred = prediction.detach().argmax(dim=1).cpu()
                    targ = target.detach().cpu()
                elif graph_data.y.shape[1] > 1:
                    pred = prediction.detach().argmax(dim=1).cpu()
                    targ = target.detach().argmax(dim=1).cpu()

                mask = graph_data.test_mask if mask_str == "test" else graph_data.val_mask if mask_str == "val" else graph_data.train_mask

                complete_predictions.append(pred[mask])
                complete_targets.append(targ[mask])

        predictions = torch.cat(complete_predictions)
        targets = torch.cat(complete_targets)

        scores = calc_test_scores(targets, predictions)

        return scores

    def visualize(self, predictions, targets, set: str, epoch: int) -> None:
        """
        This method visualizes the results of the training and test set.

        Visualizes the results of the training and test set by creating confusion matrices and plots.

        :param data_loader: Dataloader with training data
        :param epoch: Epoch number
        :return: None
        """

        self.model.eval()

        set_idx = 1 if set == 'train' else 2 if set == 'val' else 3

        self.logger.save_confusion_matrix(targets,
                                          predictions,
                                          labels=self.dataset[0].target_labels,
                                          epoch=epoch,
                                          set=f'{set_idx}_{set}')

    def save_model(self) -> None:
        """
        This method uses the `save_model` function to save the trained model to a file.
        After the model is saved, the method logs a message to the console with the path
        to the file.
        """
        path = ModelService.save_model(self.model)
        self.logger.log_model_path(model_path=path)
        print(f"Model saved to '{path}'.")

    def evaluate(self) -> None:
        """


        """
        self.model.to(self.device)
        self.loss.to(self.device)

        _, _, test_loader = self.setup_dataloaders()

        test_scores, test_results = self.test_step(test_loader, "test")
