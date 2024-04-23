"""
This module contains the Trainer class which is used to train a PyTorch model.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import pickle as pkl

from src.pipelines.model_service import ModelService
from src.utils.logger import Logger

FIG_OUTPUT_PATH: Final[Path] = Path("./data/output/eval_plot")

# create directory if it does not exist
FIG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class Trainer:
    """
    A class used to represent a Trainer for a PyTorch model.

    This class handles the training process for a PyTorch model, including setting up the
    model, loss function, and optimizer from a configuration, moving the model and loss
    function to the GPU if available, setting up the training and validation data loaders,
    and training the model for a specified number of epochs.

    Attributes:
        batch_size (int): The batch size for training.
        test_split (float): The fraction of the data to use for test.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        loss (nn.MSELoss | nn.CrossEntropyLoss | RMSELoss | RMSLELoss | ExpMSELoss): The loss function to use.
        optimizer (optim.SGD | optim.Adam): The optimizer to use.
        device (torch.device): Whether to use the CPU or the GPU.
        model (nn.Module): The PyTorch model to train.
        logger (Logger): The logger to use for logging training information.
        eval_mode (bool): Is set to True, if the evaluation function is called.
    """
    batch_size: int
    test_split: float
    epochs: int
    learning_rate: float
    loss: Union[nn.MSELoss, nn.CrossEntropyLoss]
    optimizer: Union[optim.SGD, optim.Adam]
    lr_scheduler: Union[
        optim.lr_scheduler.ReduceLROnPlateau, optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR, None]
    weight_decay: float
    device: torch.device
    model: nn.Module
    logger: Logger
    eval_mode: bool
    seed: int
    batch_shuffle: bool
    patience: int
    log_image_frequency: int

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
            momentum: float = 0,
            eval_mode: bool = False,
            seed: int = None,
            batch_shuffle: bool = False,
            patience: int = 50,
            log_image_frequency: int = 10
    ):
        """
        Creates a Trainer instance from an unpacked configuration file.
        This method sets up the loss function, model, and optimizer based on the provided
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
            momentum (float, optional): Momentum for the optimizer. Defaults to 0.
            eval_mode (bool, optional): If the model is evaluated, the validation split is set to 1.
            seed (int, optional): Seed for the random number generator.
            batch_shuffle (bool): Whether to shuffle the batches.
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 50.
            log_image_frequency (int, optional): Frequency of logging images. Defaults to 10.

        Returns:
            Trainer: A Trainer instance with the specified configuration.
        """

        # Setting up the loss
        if loss == "mse":
            loss_instance = nn.MSELoss()
        elif loss == "crossentropy":
            loss_instance = nn.CrossEntropyLoss()
        elif loss == "rmse":
            loss_instance = RMSELoss()
        elif loss == "rmsle":
            loss_instance = RMSLELoss()
        elif loss == "expmse":
            loss_instance = ExpMSELoss()
        else:
            print(f"Loss {loss} is not valid, defaulting to MSELoss")
            loss_instance = nn.MSELoss()
        print(f"Using {loss} loss function.")

        # Setting up the optimizer
        if optimizer == "adam":
            optimizer_instance = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if momentum != 0:
                print(f"Momentum {momentum} is not used since the optimizer is set to Adam")
        elif optimizer == "sgd":
            optimizer_instance = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
        else:
            print(f"Optimizer {optimizer} is not valid, defaulting to Adam")
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
                    f"Learning rate scheduler {lr_scheduler_params['scheduler']} is not valid, no scheduler is used.")
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
        self.logger = Logger()
        self.eval_mode = eval_mode
        self.seed = seed
        self.batch_shuffle = batch_shuffle
        self.patience = patience
        self.log_image_frequency = log_image_frequency
        self.logger = logger
        self.dataset = dataset
        self.test_split = test_split
        print("Trainer was successfully set up.")

    def start_training(self) -> None:
        """
        This is the entrypoint method to start the training process for the model.

        This method first moves the model and loss function to the GPU if `gpu_activated`
        is True. It then logs the trainer configuration and the model architecture. The
        method sets up the training and validation data loaders using the `setup_dataloaders`
        method. Afterward, it starts the actual training using the `train_model` method and
        logs the reason for finishing the training. After the training process is finished,
        the method closes the logger.
        """
        self.model.to(self.device)
        self.loss.to(self.device)

        # # Get string with all object variables from trainer
        # trainer_dict = vars(self).copy()
        # trainer_dict.pop("model")
        # trainer_dict.pop("logger")
        # trainer_str = str(trainer_dict).replace("'", "")
        #
        # # Get string with all object variables from dataset
        # dataset_dict = vars(self._dataset).copy()
        # dataset_str = str(dataset_dict).replace("'", "")
        #
        # # Logg object variables
        # self.logger.write_text(
        #     "config_settings/Trainer_variables", trainer_str)
        # self.logger.write_text(
        #     "config_settings/dataset_variables", dataset_str)
        # self.logger.write_model(self.model)

        # Creating training and validation data loaders from the given data
        # source
        train_loader, validation_loader = self.setup_dataloaders()

        # Perform model training
        self.logger.write_training_start()
        finish_reason = self.train_model(train_loader, validation_loader)
        self.logger.write_training_end(finish_reason)

    def setup_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Sets up the data loaders holding the training and validation datasets.

        This method splits the dataset into training and validation subsets using the dataset's
        `get_subset_indices` method. It then creates data loaders for the training and validation
        subsets with the specified batch size and without shuffling.

        Returns:
            DataLoader: The training data loader.
            DataLoader: The validation data loader.
        """
        # Splitting the dataset into training and validation sets using the dataset's subset functionality
        train_dataset, validation_dataset = random_split(self.dataset, [1 - self.test_split, self.test_split])

        # Create torch data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.batch_shuffle
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, validation_loader

    def train_model(self, train_loader: DataLoader,
                    validation_loader: DataLoader) -> str:
        """
        Trains the model for a specified number of epochs. For each epoch, the method calculates
        the training loss and validation loss, logs these losses, and saves the current state
        of the model.

        If a `KeyboardInterrupt` is raised during training, the method catches it and sets the
        finish reason to `"Training interrupted by user"`. If the training finishes without
        interruption, the finish reason is set to `"Training finished normally"`.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            validation_loader (DataLoader): DataLoader for the validation set.

        Returns:
            str: The reason the training ended.
        """
        # Setup for early stopping
        min_loss = float('inf')
        cur_patience = 0

        finish_reason = "Training terminated before training loop ran through."
        for epoch in tqdm(range(self.epochs)):
            try:
                train_loss = self.calculate_train_loss(
                    train_loader)

                # Logging loss and charts of results
                self.logger.log_training_loss(train_loss, epoch)

                validation_loss = self.calculate_validation_loss(
                    validation_loader)

                # Logging loss and charts of results
                self.logger.log_validation_loss(validation_loss, epoch)

                # Logging learning rate (getter-function only works with torch2.2 or higher)
                if self.lr_scheduler is not None:
                    try:
                        self.logger.log_lr(lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    except AttributeError:
                        self.logger.log_lr(lr=self.lr_scheduler.optimizer.param_groups[0]['lr'], epoch=epoch)

                # Early stopping
                if min_loss > validation_loss:
                    min_loss = validation_loss
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

    def calculate_train_loss(self, train_loader) -> float:
        """
        Calculates the training loss for the model. This method is called during each epoch.

        This method iterates over each batch in the training loader. For each batch, it
        resets the optimizer, calculates the loss between the predictions and the actual
        targets, performs backpropagation, and updates the model's parameters.
        The method accumulates the total training loss and returns the average training
        loss per batch.

        If `gpu_activated` is True, the method moves the batch to the GPU before computing the
        predictions and loss.

        Returns:
            float: The average training loss per batch.
        """
        self.model.train()
        train_loss: float = 0
        step_count: int = 0

        loder_len = len(train_loader)

        # create an array to store the predictions and targets of all samples
        #samples = len(train_loader.dataset)
        #prediction_len = train_loader.dataset.dataset.decoder_target_length
        #dim = train_loader.dataset.dataset.decoder_dimensions
        #results = np.zeros((2, samples, prediction_len, dim))

        for input_data, target in train_loader:

            # Reset optimizer
            self.optimizer.zero_grad()

            input_data = input_data.to(self.device)
            target = target.to(self.device)

            prediction = self.model.forward(input_data)
            loss = self.loss(prediction, target.float())

            # Safe prediction and target for visualization
            #start_idx = step_count * self.batch_size
            #end_idx = start_idx + self.batch_size
            #results[0, start_idx:end_idx, :, :] = prediction.detach().cpu()
            #results[1, start_idx:end_idx, :, :] = target.detach().cpu()

            loss.backward()

            self.optimizer.step()

            # Step for ReduceLROnPlateau schedule is done with validation loss
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler,
                                                                optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

            train_loss += loss.item()
            step_count += 1

            if step_count % 50 == 0:
                print(
                    f'Batch {step_count}/{loder_len} loss: {round(loss.item(), 2)}')

        loss = train_loss / step_count

        return loss

    def calculate_validation_loss(self, validation_loader) -> float:
        """
        Calculates the validation loss for the model. This method is called during each epoch.

        This method iterates over each batch in the validation loader, computes the model's
        predictions for the batch, calculates the loss between the predictions and the actual
        targets, and accumulates the total validation loss. The method returns the average
        validation loss per batch.

        If `gpu_activated` is True, the method moves the batch to the GPU before computing the
        predictions and loss.

        Returns:
            float: The average validation loss per batch.
        """
        self.model.eval()
        validation_loss: float = 0
        step_count: int = 0

        # create an array to store the predictions and targets of all samples
        #samples = len(validation_loader.dataset)
        #prediction_len = validation_loader.dataset.dataset.decoder_target_length
        #dim = validation_loader.dataset.dataset.decoder_dimensions
        #results = np.zeros((2, samples, prediction_len, dim))

        loder_len = len(validation_loader)

        with torch.no_grad():
            for input_data, target in validation_loader:

                input_data = input_data.to(self.device)
                target = target.to(self.device)

                prediction = self.model.forward(input_data)
                loss = self.loss(prediction, target.float())

                #start_idx = step_count * self.batch_size
                #end_idx = start_idx + self.batch_size
                #results[0, start_idx:end_idx, :, :] = prediction.cpu()
                #results[1, start_idx:end_idx, :, :] = target.cpu()

                if self.lr_scheduler is not None and isinstance(self.lr_scheduler,
                                                                optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(loss)

                validation_loss += loss.sum().item()
                step_count += 1

                if step_count % 10 == 0:
                    print(
                        f'Batch {step_count}/{loder_len} loss: {loss.item()}')

        loss = validation_loss / step_count

        return loss

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

        loss, results = self.calculate_validation_loss(validation_loader)

        predictions = results[0]
        targets = results[1]

        plot = plot_evaluation(targets, predictions)
        plot.show()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = Path(FIG_OUTPUT_PATH, f'plot{now}.png')

        plot.savefig(path)

        Logger.log_text(f"Evaluation plot saved to '{path}'.")

        print(loss)
