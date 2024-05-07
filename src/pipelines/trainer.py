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
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
            test_split: float,
            epochs: int,
            learning_rate: float,
            device: torch.device,
            logger: Logger,
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
            if loss != 'nll':
                y_labels = torch.cat([dataset[i].y.argmax(dim=1) for i in range(len(dataset))]).numpy()
                label_classes = torch.unique(dataset[0].y.argmax(dim=1)).numpy()
            else:
                y_labels = torch.cat([dataset[i].y for i in range(len(dataset))]).numpy()
                label_classes = torch.unique(dataset[0].y).numpy()
            class_weights = compute_class_weight('balanced',
                                                 classes=label_classes,
                                                 y=y_labels)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
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
        dataloader = self.setup_dataloaders()

        # Perform model training
        self.logger.write_training_start()
        finish_reason = self.train_model(dataloader)
        self.logger.write_training_end(finish_reason)

    def setup_dataloaders(self) -> DataLoader:
        """
        Sets up the data loaders holding the graph dataset.

        Returns:
            DataLoader: The training and test data loader.
        """

        # Create torch data loaders
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

        return dataloader

    def train_model(self, dataloader: DataLoader) -> str:
        """
        Trains the model for a specified number of epochs. For each epoch, the method calculates
        the training loss and validation loss, logs these losses, and saves the current state
        of the model.

        If a `KeyboardInterrupt` is raised during training, the method catches it and sets the
        finish reason to `"Training interrupted by user"`. If the training finishes without
        interruption, the finish reason is set to `"Training finished normally"`.

        Args:
            dataloader (DataLoader): DataLoader for the training and the test set.

        Returns:
            str: The reason the training ended.
        """
        # Setup for early stopping
        min_loss = float('inf')
        cur_patience = 0

        finish_reason = "Training terminated before training loop ran through."
        for epoch in tqdm(range(self.epochs)):
            try:
                train_loss, test_loss = self.train_step(dataloader)

                # Logging loss of results
                self.logger.log_training_loss(train_loss, epoch)
                self.logger.log_test_loss(test_loss, epoch)

                # Calculating and logging evaluation results
                if epoch % self.log_image_frequency == 0:
                    validation_score = self.test_step(dataloader, epoch)
                    self.logger.log_accuracy_score(validation_score, epoch)

                # Logging learning rate (getter-function only works with torch2.2 or higher)
                if self.lr_scheduler is not None:
                    try:
                        self.logger.log_lr(lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    except AttributeError:
                        self.logger.log_lr(lr=self.lr_scheduler.optimizer.param_groups[0]['lr'], epoch=epoch)

                # Early stopping
                if min_loss > test_loss:
                    min_loss = test_loss
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
        total_test_loss: float = 0
        step_count: int = 0

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

            # Get test loss
            test_loss = self.loss(prediction[graph_data.test_mask], target[graph_data.test_mask])

            total_train_loss += train_loss.item()
            total_test_loss += test_loss.item()
            step_count += 1

            # Step for ReduceLROnPlateau schedule is done with validation loss
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler,
                                                                optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

        total_train_loss = total_train_loss / step_count
        total_test_loss = total_test_loss / step_count

        return total_train_loss, total_test_loss

    def test_step(self, validation_loader, epoch: int) -> float:
        """
        Calculates the target metric for the test set and generates visualisations for the train and the test set. This method is called in the frequency given in the config.

        This method iterates over each batch in the dataloader, computes the model's
        predictions for the batch, calculates the accuracy between the predictions and the actual
        targets, and accumulates the total accuracy. The method returns the average
        accuracy over all batches.

        Returns:
            float: The accuracy over all batches.
        """
        self.model.eval()
        total_accuracy: float = 0
        step_count: int = 0

        loder_len = len(validation_loader)

        complete_predictions = []
        complete_targets = []
        complete_train_mask = []
        complete_test_mask = []

        with torch.no_grad():
            for graph_data in validation_loader:

                if isinstance(graph_data.x, torch.Tensor):
                    input_graph_feature = graph_data.x.to(self.device)
                else:
                    input_graph_feature = graph_data.x
                input_graph_edge_index = graph_data.edge_index.to(self.device)
                target = graph_data.y.float().to(self.device)

                prediction = self.model.forward(input_graph_feature, input_graph_edge_index)

                if len(graph_data.y.shape) == 1:
                    pred = prediction.cpu().argmax(dim=1)
                    targ = target.cpu()
                elif graph_data.y.shape[1] > 1:
                    pred = prediction.argmax(dim=1).cpu()
                    targ = target.argmax(dim=1).cpu()

                complete_predictions.append(pred)
                complete_targets.append(targ)
                complete_train_mask.append(graph_data.train_mask)
                complete_test_mask.append(graph_data.test_mask)

                accuracy = accuracy_score(targ[graph_data.test_mask],
                                          pred[graph_data.test_mask])


                total_accuracy += accuracy
                step_count += 1

                if step_count % 10 == 0:
                    print(
                        f'[TRAINER]: Batch {step_count}/{loder_len} accuracy: {accuracy}')

        total_accuracy = total_accuracy / step_count

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)
        complete_train_mask = torch.cat(complete_train_mask)
        complete_test_mask = torch.cat(complete_test_mask)

        self.logger.save_confusion_matrix(complete_targets[complete_train_mask],
                                          complete_predictions[complete_train_mask],
                                          labels=graph_data.target_labels,
                                          epoch=epoch,
                                          set='1_Train')

        self.logger.save_confusion_matrix(complete_targets[complete_test_mask],
                                          complete_predictions[complete_test_mask],
                                          labels=graph_data.target_labels,
                                          epoch=epoch,
                                          set='2_Test')

        return total_accuracy

    def save_model(self) -> None:
        """
        This method uses the `save_model` function to save the trained model to a file.
        After the model is saved, the method logs a message to the console with the path
        to the file.
        """
        path = ModelService.save_model(self.model)
        self.logger.log_model_path(model_path=path)
        Logger.log_text(f"Model saved to '{path}'.")

    def evaluate(self) -> None:
        """


        """
        self.model.to(self.device)
        self.loss.to(self.device)

        # Creating training and validation data loaders from the given data
        # source
        self.validation_split = 1
        train_loader, validation_loader = self.setup_dataloaders()

        loss, results = self.test_step(validation_loader)

        predictions = results[0]
        targets = results[1]

        plot = plot_evaluation(targets, predictions)
        plot.show()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = Path(FIG_OUTPUT_PATH, f'plot{now}.png')

        plot.savefig(path)

        Logger.log_text(f"Evaluation plot saved to '{path}'.")

        print(loss)
