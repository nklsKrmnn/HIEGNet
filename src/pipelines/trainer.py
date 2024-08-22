"""
This module contains the Trainer class which is used to train a PyTorch model.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.evaluation.loss_functions import WeightedMSELoss
from src.evaluation.test_scores import calc_test_scores
from src.utils.model_service import ModelService
from src.utils.logger import Logger
from src.evaluation.vis_graph import visualize_graph

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
            log_image_frequency: int = 10,
            reported_set: str = "val"
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
            reported_set (str, optional): The set to report the performance measures on. Defaults to "test".

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
        elif loss == "weighted_mse":
            if class_weights_tensor is None:
                print(f"[TRAINER]: Class weights are not available, defaulting to MSELoss")
                loss_instance = nn.MSELoss()
            else:
                loss_instance = WeightedMSELoss(class_weights_tensor)
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
        self.reported_set = reported_set
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
        train_dataset = Subset(self.dataset, train_indices)
        validation_dataset = Subset(self.dataset, validation_indices)
        test_dateset = Subset(self.dataset, test_indices)

        # Check that dataset to report is not empty
        if self.reported_set == "test" and test_indices == []:
            raise ValueError("No test set to report performance on.")
        if self.reported_set == "val" and validation_indices == []:
            raise ValueError("No validation set to report performance on.")
        if train_indices == []:
            raise ValueError("No training set to train on.")

        # Create torch data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size) if validation_indices != [] else None
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

                if validation_loader is not None:
                    # Calculating validation loss if loader exists
                    val_loss, val_results = self.validation_step(validation_loader)
                    self.logger.log_loss(val_loss, epoch, "2_validation")
                    early_stopping_loss = val_loss
                elif train_loader is not None and self.reported_set == "test":
                    # Calculating test loss if loader exists, and we want to report it
                    test_loss, test_results = self.validation_step(test_loader)
                    self.logger.log_loss(test_loss, epoch, "3_test")
                    early_stopping_loss = test_loss

                # Calculate test scores on set we want to report
                if test_loader is not None and self.reported_set == "test":
                    test_scores, test_results = self.test_step(test_loader, "test")
                elif validation_loader is not None and self.reported_set == 'val':
                    test_scores = calc_test_scores(val_results[1], val_results[0])
                elif train_loader is not None and self.reported_set == 'train':
                    test_scores = calc_test_scores(train_results[1], train_results[0])
                else:
                    raise ValueError("No valid set to report performance on.")

                for score, score_dict in test_scores.items():
                    for class_label, value in score_dict.items():
                        self.logger.log_test_score(value, epoch, class_label, score)

                if epoch % self.log_image_frequency == 0:
                    self.visualize(train_results[0], train_results[1], 'train', epoch)
                    if validation_loader is not None:
                        self.visualize(val_results[0], val_results[1], "val", epoch)
                    if test_loader is not None and self.reported_set == "test":
                        self.visualize(test_results[0], test_results[1], 'test', epoch)

                # Step for ReduceLROnPlateau schedule is done with validation loss
                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler,
                                                                    optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

                # Logging learning rate (getter-function only works with torch2.2 or higher)
                if self.lr_scheduler is not None:
                    try:
                        self.logger.log_lr(lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    except AttributeError:
                        self.logger.log_lr(lr=self.lr_scheduler.optimizer.param_groups[0]['lr'], epoch=epoch)

                # Early stopping
                if min_loss > early_stopping_loss:
                    min_loss = early_stopping_loss
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

    def train_step(self, dataloader) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
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

            pred, targ, loss = self.calc_batch(graph_data, graph_data.train_mask)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            step_count += 1

            complete_predictions.append(pred[graph_data.train_mask.detach().cpu()])
            complete_targets.append(targ[graph_data.train_mask.detach().cpu()])

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_train_loss = total_train_loss / step_count

        return total_train_loss, (complete_predictions, complete_targets)

    def calc_batch(self, graph_data, mask, return_softmax:bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        graph_data = graph_data.to(self.device)
        # Determine model inputs as send to device
        if not isinstance(graph_data[0], HeteroData):
            input_graph_feature = graph_data.x if isinstance(graph_data.x, torch.Tensor) else graph_data.x
        else:
            input_graph_feature = graph_data.x_dict
        input_graph_edge_index = graph_data.edge_index if not isinstance(graph_data[0],
                                                                         HeteroData) else graph_data.edge_index_dict
        input_graph_attr = graph_data.edge_attr if not isinstance(graph_data[0],
                                                                  HeteroData) else graph_data.edge_attr_dict
        target = graph_data.y.long() if isinstance(self.loss, nn.NLLLoss) else graph_data.y

        prediction = self.model.forward(input_graph_feature, input_graph_edge_index, input_graph_attr)
        loss = self.loss(prediction[mask], target[mask])

        if (len(graph_data.y.shape) == 1) or return_softmax:
            pred = prediction.detach().cpu()
            targ = target.detach().cpu()
        elif graph_data.y.shape[1] > 1:
            pred = prediction.detach().argmax(dim=1).cpu()
            targ = target.detach().argmax(dim=1).cpu()
        else:
            raise ValueError("Target shape is not valid.")

        return pred, targ, loss

    def validation_step(self, validation_loader) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
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
                pred, targ, val_loss = self.calc_batch(graph_data, graph_data.val_mask)

                total_val_loss += val_loss.item()
                step_count += 1

                complete_predictions.append(pred[graph_data.val_mask.detach().cpu()])
                complete_targets.append(targ[graph_data.val_mask.detach().cpu()])

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        total_val_loss = total_val_loss / step_count

        return total_val_loss, (complete_predictions, complete_targets)

    def test_step(self,
                  test_loader,
                  mask_str: str = "test",
                  return_softmax: bool = False) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
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
                mask = graph_data.test_mask if mask_str == "test" else graph_data.val_mask if mask_str == "val" else graph_data.train_mask
                pred, targ, _ = self.calc_batch(graph_data, mask, return_softmax=True)

                complete_predictions.append(pred[mask.detach().cpu()])
                complete_targets.append(targ[mask.detach().cpu()])

        complete_predictions = torch.cat(complete_predictions)
        complete_targets = torch.cat(complete_targets)

        if graph_data.y.shape[1] > 1:
            pred = complete_predictions.argmax(dim=1)
            targ = complete_targets.argmax(dim=1)
        else:
            raise ValueError("Target shape is not valid.")

        # Calculate test scores
        scores = calc_test_scores(targ, pred)

        softmax_results = (complete_predictions, complete_targets)
        results = softmax_results if return_softmax else (pred, targ)

        return scores, results

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

        # Transform to np.array
        predictions = predictions.numpy()
        targets = targets.numpy()

        use_continuous_matrix = (isinstance(self.loss, nn.MSELoss) or isinstance(self.loss, WeightedMSELoss))

        self.logger.save_confusion_matrix(targets,
                                          predictions,
                                          labels=self.dataset[0].target_labels,
                                          epoch=epoch,
                                          continuous=use_continuous_matrix,
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

    def evaluate(self, parameters: dict[str, dict]) -> None:
        # TODO: Docstring
        """


        """
        self.model.to(self.device)
        self.loss.to(self.device)

        self.model.eval()

        _, _, test_loader = self.setup_dataloaders()

        if test_loader is None:
            raise ValueError("No test set to evaluate on.")

        test_scores, test_results = self.test_step(test_loader, "test", return_softmax=True)

        # Unstack scores
        test_scores = {f'{metric}_{score}': value for metric, score_dict in test_scores.items() for score, value in score_dict.items()}

        # Save test scores and parameters
        params = {f'params_{param_set}_{key}': value for param_set, param_dict in parameters.items() for key, value in param_dict.items()}
        test_scores.update(params)
        test_scores['name'] = self.logger.name.split('/')[-1]
        test_scores = pd.DataFrame([test_scores])
        try:
            scores_file = pd.read_csv("./data/output/test_scores.csv") # TODO: put as constant
        except:
            scores_file = pd.DataFrame({})

        scores_file = pd.concat([scores_file, test_scores], ignore_index=True)
        scores_file.to_csv("./data/output/test_scores.csv", index=False)


        # Dump softmax scores (test results) as csv file
        df_test_predictions = pd.DataFrame(test_results[0].numpy())
        df_test_predictions.columns = [f"prediction_{label}" for label in self.dataset[0].target_labels]
        df_test_targets = pd.DataFrame(test_results[1].numpy())
        df_test_targets.columns = [f"target_{label}" for label in self.dataset[0].target_labels]
        df_test_results = pd.concat([df_test_predictions, df_test_targets], axis=1)

        # Get images for test set
        glomeruli_indices = []
        for batch in test_loader:
            glomeruli_indices.append(batch["glom_indices"][batch["test_mask"]])
        df_test_results["glomerulus_index"] = torch.cat(glomeruli_indices).numpy()

        df_test_results.to_csv(f"./data/output/test_results.csv", index=False)

        if test_loader.batch_size == 1:
            for i, batch in enumerate(test_loader):
                coordinates = batch["coords"].numpy()
                sparse_matrix = batch[('glomeruli', 'to', 'glomeruli')].edge_index.numpy()
                target_classes = batch['y']
                with torch.no_grad():
                    prediction, _, _ = self.calc_batch(batch, torch.ones_like(batch.test_mask))

                target_classes = target_classes.numpy().argmax(axis=1)
                predicted_classes = prediction.numpy()

                class_labels = self.dataset[0].target_labels
                fig = visualize_graph(coordinates,
                                      sparse_matrix,
                                      target_classes,
                                      predicted_classes,
                                      class_labels,
                                      batch.train_mask,
                                      batch.test_mask)
                self.logger.save_figure(fig, "graph", i)


