"""
This module contains the Trainer class which is used to train a PyTorch model.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from src.evaluation.test_scores import calc_test_scores
from src.pipelines.trainer import Trainer
from src.utils.model_service import ModelService
from src.utils.logger import Logger

FIG_OUTPUT_PATH: Final[Path] = Path("./data/output/eval_plot")

# create directory if it does not exist
FIG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageTrainer(Trainer):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def train_model_old(self, train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader) -> str:
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
                if len(validation_loader) > 0:
                    val_loss, val_results = self.validation_step(validation_loader)
                    self.logger.log_loss(val_loss, epoch, "2_validation")

                if test_loader is not None and self.reported_set == "test":
                    test_scores, test_results = self.test_step(test_loader, return_softmax=False)
                elif validation_loader is not None and self.reported_set == 'val':
                    test_results = (val_results[0], val_results[1])
                    test_scores = calc_test_scores(val_results[1], val_results[0])
                elif train_loader is not None and self.reported_set == 'train':
                    test_results = (train_results[0], train_results[1])
                    test_scores = calc_test_scores(train_results[1], train_results[0])
                else:
                    raise ValueError("No valid set to report performance on.")

                for score, score_dict in test_scores.items():
                    for class_label, value in score_dict.items():
                        self.logger.log_test_score(value, epoch, class_label, score)

                if epoch % self.log_image_frequency == 0:
                    self.visualize(train_results[0], train_results[1], 'train', epoch)
                    if len(validation_loader) > 0:
                        self.visualize(val_results[0], val_results[1], "val", epoch)
                    if len(test_loader) > 0:
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

    def train_step_old(self, dataloader) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculates the training loss for the model. This method is called during each epoch.

        This method iterates over each batch in the training loader. For each batch, it resets the optimizer,
        calculates the loss between the predictions and the actual targets, performs backpropagation, and updates the
        model's parameters. The forward function is computed for the whole dataset. The train and test mask are used
        to separate the dataset into training and test data. The method accumulates the total training loss and
        returns the average training loss per batch.

        Returns:
            float: The average training loss per batch.
        """
        self.model.train()
        total_train_loss: float = 0
        step_count: int = 0
        complete_predictions = []
        complete_targets = []

        for images, labels in dataloader:

            # Reset optimizer
            self.optimizer.zero_grad()

            # Calc batch
            pred, targ, train_loss = self.calc_batch(images, labels)

            # Backpropagation
            train_loss.backward()
            self.optimizer.step()

            total_train_loss += train_loss.item()
            step_count += 1

            complete_predictions.append(pred)
            complete_targets.append(targ)

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_train_loss = total_train_loss / step_count

        return total_train_loss, (complete_predictions, complete_targets)

    def calc_batch(self, data, return_softmax: bool = False, mask_str:bool=None) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the predictions and targets for a batch of data.

        This method calculates the predictions and targets for a batch of data. The method is called during each epoch
        for the training and validation data. The method returns the predictions and targets for the batch.

        Args:
            graph_data (DataLoader): DataLoader for the training and validation graphs.
            mask_str (str): The mask to use for separating the dataset into training and validation data.
            return_softmax (bool): Whether to return the softmax results.

        Returns:
            torch.Tensor: The predictions for the batch.
            torch.Tensor: The targets for the batch.
        """
        image, labels = data[0].to(self.device), data[1].to(self.device)

        predictions = self.model.forward(image)
        loss = self.loss(predictions, labels)

        if len(labels.shape) == 1 or return_softmax:
            pred = predictions.detach().cpu()
            targ = labels.detach().cpu()
        elif labels.shape[1] > 1:
            pred = predictions.detach().argmax(dim=1).cpu()
            targ = labels.detach().argmax(dim=1).cpu()
        else:
            raise ValueError("Target shape is not valid.")

        return pred, targ, loss

    def validation_step_old(self, validation_loader) -> float:
        """
        Calculates the target metric for the test set and generates visualisations for the train and the test set. This method is called in the frequency given in the config.

        This method iterates over each batch in the dataloader, computes the model's
        predictions for the batch, calculates the accuracy between the predictions and the actual
        targets, and accumulates the total accuracy. The method returns the average
        accuracy over all batches.

        Returns:
            float: The loss over all batches.
        """
        self.model.eval()
        complete_predictions = []
        complete_targets = []
        total_val_loss: float = 0
        step_count: int = 0

        with torch.no_grad():
            for images, labels in validation_loader:

                # Reset optimizer
                self.optimizer.zero_grad()

                # Calc batch
                pred, targ, val_loss = self.calc_batch(images, labels)

                total_val_loss += val_loss.item()
                step_count += 1

                # Step for ReduceLROnPlateau schedule is done with validation loss
                if self.lr_scheduler is not None and isinstance(self.lr_scheduler,
                                                                    optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

                complete_predictions.append(pred)
                complete_targets.append(targ)

        # Calc test scores
        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_val_loss = total_val_loss / step_count

        return total_val_loss, (complete_predictions, complete_targets)

    def test_step_old(self, test_loader,
                  return_softmax: bool = False) -> dict[str, dict[str, float]]:
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
            for images, labels in test_loader:

                # Reset optimizer
                self.optimizer.zero_grad()

                # Calc batch
                pred, targ, _ = self.calc_batch(images, labels, return_softmax=return_softmax)

                complete_predictions.append(pred)
                complete_targets.append(targ)


        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)


        test_scores = calc_test_scores(pred, targ)

        # TODO: Fix to return softmax results
        softmax_results = (complete_predictions, complete_targets)
        results = softmax_results if return_softmax else (pred, targ)

        return test_scores, results

    def visualize_old(self, predictions, targets, set: str, epoch: int) -> None:
        """
        This method visualizes the results of the training and test set.

        Visualizes the results of the training and test set by creating confusion matrices and plots.

        :param data_loader: Dataloader with training data
        :param epoch: Epoch number
        :return: None
        """
        set_idx = 1 if set == 'train' else 2 if set == 'val' else 3

        self.logger.save_confusion_matrix(targets,
                                          predictions,
                                          labels=self.dataset.target_labels,
                                          epoch=epoch,
                                          set=f'{set_idx}_{set}')

    def save_model_old(self) -> None:
        """
        This method uses the `save_model` function to save the trained model to a file.
        After the model is saved, the method logs a message to the console with the path
        to the file.
        """
        path = ModelService.save_model(self.model)
        self.logger.log_model_path(model_path=path)
        print(f"Model saved to '{path}'.")

    def evaluate(self, parameters: dict[str, dict]) -> None:
        """


        """
        self.model.to(self.device)
        self.loss.to(self.device)

        self.model.eval()

        _, _, test_loader = self.setup_dataloaders()

        if test_loader is None:
            raise ValueError("No test set to evaluate on.")

        test_scores, test_results = self.test_step(test_loader, return_softmax=True)

        # Unstack scores
        test_scores = {f'{metric}_{score}': value for metric, score_dict in test_scores.items() for score, value in
                       score_dict.items()}

        # Save test scores and parameters
        params = {f'params_{param_set}_{key}': value for param_set, param_dict in parameters.items() for key, value in
                  param_dict.items()}
        test_scores.update(params)
        test_scores['name'] = self.logger.name.split('/')[-1]
        test_scores = pd.DataFrame([test_scores])
        try:
            scores_file = pd.read_csv("./data/output/test_scores_images.csv")  # TODO: put as constant
        except:
            scores_file = pd.DataFrame({})

        scores_file = pd.concat([scores_file, test_scores], ignore_index=True)
        scores_file.to_csv("./data/output/test_scores_image.csv", index=False)



