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



