"""
This module contains the Logger class which is used to log training information.
"""
import inspect
import io
import os
from datetime import datetime
from typing import Optional
import pandas as pd
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.utils.vis_training import plot_confusion_matrix, plot_continous_confussion_matrix

first_name_logging = True


class Logger():
    """
    A class used to log training information.

    This class handles logging of training information to the console and to a TensorBoard log file.
    It logs the start and end times of the training, the training and validation losses for each
    epoch, and the model architecture. Additionally, it handles closing the logger after training is
    finished.
    """

    def __init__(self, name: str = "") -> None:
        """
        Initializes the Logger.

        This init method sets up the TensorBoard writer with the name of the target directory set
        to the current date and time. It also initializes the training start time to None.
        """
        current_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._summary_writer = SummaryWriter(f"runs/{current_time_string}_{name}")
        self._training_start: Optional[datetime] = None

    def write_text(self, tag: str, text: str) -> None:
        """
        Writes a custom text to a custom tag in the TensorBoard log file.

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        self._summary_writer.add_text(tag, text)

    def write_config(self, config: dict) -> None:
        """
        Logs the configuration of the training.

        This method writes for each first level item in the config a text of its contents with the dict key as subtag.

        Args:
            config (dict): The configuration of the training.
        """
        for key, value in config.items():
            self.write_text(f'config_{key}', str(value))

    def write_model(self, model: nn.Module) -> None:
        """
        Writes the model architecture formatted as text to the TensorBoard log file.

        Args:
            model (nn.Module): The model.
        """
        self._summary_writer.add_text("model/model_class", str(model))

    def write_training_start(self) -> None:
        """
        Logs the start time of the training.

        This method writes a message to the console, sets the training start time to the current
        time, and writes the start time to the TensorBoard log file.
        """
        tqdm.write("[LOGGER]: Training was started.")

        self._training_start = datetime.now()
        self._summary_writer.add_text(
            "training_duration/start_time", str(self._training_start))

    def write_training_end(self, reason: str) -> None:
        """
        Logs the end time of the training and the reason the training finished.

        This method calculates the training duration, writes a message to the console,
        and writes the end time, training duration, and finish reason to the TensorBoard log file.
        It also closes the TensorBoard writer.

        Args:
            reason (str): The reason the training finished.
        """
        training_end = datetime.now()

        if self._training_start is None:
            # Set training duration to None if training start is None (if
            # log_training_start was not called)
            training_duration = None
        else:
            training_duration = training_end - self._training_start
        tqdm.write(
            f"[LOGGER]: Training finished with a runtime of {training_duration}. Finish reason: {reason}")

        self._summary_writer.add_text(
            "training_duration/end_time", str(training_end))
        self._summary_writer.add_text("training_duration/duration",
                                      str(training_duration))
        self._summary_writer.add_text("training_duration/reason", reason)

        tqdm.write("[LOGGER]: Closing logger.")
        self._summary_writer.close()

    def log_loss(self, value: float, epoch: int, set: str = 'train') -> None:
        """
        Logs the loss for an epoch.

        This method writes a message to the console and writes the loss to the
        TensorBoard log file.

        Args:
            value (float): The loss.
            epoch (int): The epoch number.
            set (str): The set for which the loss is logged. Defaults to 'train'.
        """
        tqdm.write(f"[LOGGER]: Epoch {epoch}: {set.split('_')[1]} Loss = {value}")
        self._summary_writer.add_scalar(f"loss/{set}", value, epoch)

    def log_training_loss(self, value: float, epoch: int):
        """
        Logs the training loss for an epoch.

        This method writes a message to the console and writes the training loss to the
        TensorBoard log file.

        Args:
            value (float): The training loss.
            epoch (int): The epoch number.
        """
        tqdm.write(f"[LOGGER]: Epoch {epoch}: Training Loss = {value}")
        self._summary_writer.add_scalar("loss/1_train", value, epoch)

    def log_test_loss(self, value: float, epoch: int) -> None:
        """
        Logs the test loss for an epoch.

        This method writes a message to the console and writes the test loss to the
        TensorBoard log file.

        Args:
            value (float): The test loss.
            epoch (int): The epoch number.
        """
        tqdm.write(f"[LOGGER]: Epoch {epoch}: Test Loss = {value}")
        self._summary_writer.add_scalar("loss/2_test", value, epoch)

    def log_test_score(self, value: float, epoch: int, class_label: str = '0_total', score: str = "Accuracy") -> None:
        """
        Logs the accuracy for an epoch.

        This method writes a message to the console and writes the accuracy to the
        TensorBoard log file.

        Args:
            value (float): The accuracy.
            epoch (int): The epoch number.
            class_label (str): The set for which the accuracy is logged. Defaults to 'val'.
        """
        self._summary_writer.add_scalar(f"{score}/{class_label}", value, epoch)

    def log_lr(self, lr: float, epoch: int) -> None:
        """
        Logs the learning rate for an epoch.

        This method writes a message to the console and writes the validation loss to the
        TensorBoard log file.

        Args:
            value (float): The validation loss.
            epoch (int): The epoch number.
        """
        self._summary_writer.add_scalar("lr", lr, epoch)

    def save_confusion_matrix(self,
                              y_true: np.array,
                              y_pred: np.array,
                              labels: list[str],
                              epoch: int,
                              continuous: bool = False,
                              set='Test'):
        """
        Saves a confusion matrix to the TensorBoard log file.
        Args:
            targets (np.array): Targets of the model.
            predictions (np.array): Predictions of the model.
            labels (list[str]): List of class labels.
            epoch (int): Epoch, in which the predictions were made.
            name (str): Name of the chart. Defaults to "validation_set".

        Returns: None
        """
        title = f'Confusion Matrix: {set} Set'
        cmap = 'Blues'

        if not continuous:
            fig = plot_confusion_matrix(y_true, y_pred, labels, title, cmap)
        else:
            fig = plot_continous_confussion_matrix(y_true, y_pred, labels, title, cmap)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Save image in Logger
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        self._summary_writer.add_image(f"image/cm_{set}", image, epoch)

        print(f"[Logger]: Chart for {set} set saved.")
        plt.close('all')

    def log_model_path(self, model_path: str) -> None:
        """
        Logs the path of the saved model.
        Args:
            model_path (str): Path of the saved model.

        Returns: None

        """
        global first_name_logging

        if first_name_logging:
            self._summary_writer.add_text("model/mode_path", model_path)
            first_name_logging = False


class FoldLogger(Logger):
    """
    A class used to log training information for each run of a cross validation.

    It logs the start and end times of the training, the training and validation losses for each
    epoch, and the model architecture. Additionally, it handles closing the logger after training is
    finished. This class inherits from the Logger class and extends it by adding lists attributes to store the
    values logged during training to be accessible after training is finished.
    """

    def __init__(self, fold: int, current_time_string, name: str = "") -> None:

        self.start_time_str = current_time_string
        self._summary_writer = SummaryWriter(f"runs/{current_time_string}_{name}_fold{fold}")
        self._training_start: Optional[datetime] = None

        self.train_loss = {}
        self.val_loss = {}
        self.test_loss = {}
        self.scores = {}
        self.text = {}

    def write_text(self, tag: str, text: str) -> None:
        """
        Writes a custom text to a custom tag in the TensorBoard log file and as class attribute.

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        super().write_text(tag, text)
        self.text[tag] = text

    def write_config(self, config: dict) -> None:
        """
        Logs the configuration of the training.

        This method writes for each first level item in the config a text of its contents with the dict key as subtag.

        Args:
            config (dict): The configuration of the training.
        """
        config = copy.deepcopy(config)
        super().write_config(config)
        for key, value in config.items():
            self.text[f'config_{key}'] = str(value)

    def write_model(self, model: nn.Module) -> None:
        """
        Writes the model architecture formatted as text to the TensorBoard log file and as class attribute.

        Args:
            model (nn.Module): The model.
        """
        super().write_model(model)
        self.model = str(model)

    def log_loss(self, value: float, epoch: int, set: str = 'train') -> None:
        """
        Logs the loss for an epoch.

        This method writes a message to the console and writes the loss to the
        TensorBoard log file.

        Args:
            value (float): The loss.
            epoch (int): The epoch number.
            set (str): The set for which the loss is logged. Defaults to 'train'.
        """
        super().log_loss(value, epoch, set)
        if set == '1_train':
            self.train_loss[epoch] = value
        elif set == '2_validation':
            self.val_loss[epoch] = value
        elif set == '3_test':
            self.test_loss[epoch] = value

    # TODO remove bellow loss logging methods
    def log_training_loss(self, value: float, epoch: int):
        """
        Logs the training loss for an epoch.

        This method writes a message to the console, writes the train loss to the
        TensorBoard log file and into a list as class attribute.

        Args:
            value (float): The training loss.
            epoch (int): The epoch number.
        """
        super().log_training_loss(value, epoch)
        self.train_loss[epoch] = value

    def log_test_loss(self, value: float, epoch: int) -> None:
        """
        Logs the test loss for an epoch.

        This method writes a message to the console, writes the test loss to the
        TensorBoard log file and into a list as class attribute.

        Args:
            value (float): The test loss.
            epoch (int): The epoch number.
        """
        super().log_test_loss(value, epoch)
        self.test_loss[epoch] = value

    def log_test_score(self, value: float, epoch: int, class_label: str = '0_total', score: str = "Accuracy") -> None:
        """
        Logs the accuracy for an epoch.

        This method writes a message to the console, writes the accuracy to the
        TensorBoard log file and into a list as class attribute.

        Args:
            value (float): The accuracy.
            epoch (int): The epoch number.
            class_label (str): The set for which the accuracy is logged. Defaults to 'val'.
        """
        super().log_test_score(value, epoch, class_label, score)
        if score not in self.scores.keys():
            self.scores[score] = {}
        if class_label not in self.scores[score].keys():
            self.scores[score][class_label] = {}
        self.scores[score][class_label][epoch] = value

    def log_model_path(self, model_path: str) -> None:
        """
        Logs the path of the saved model.
        Args:
            model_path (str): Path of the saved model.

        Returns: None

        """
        global first_name_logging

        if first_name_logging:
            self._summary_writer.add_text("model/mode_path", model_path)
            self.text["model/mode_path"] = model_path
            first_name_logging = False


class CrossValLogger():
    """
    A class used to summarize the training information for all folds of a cross validation.

    It copies text information from a logger and calculates the mean values of the training
    loss, test loss, and accuracy of multiple loggers
    """

    def __init__(self, n_folds: int, name: str = "") -> None:

        self.start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.fold_logger = [FoldLogger(fold, self.start_time_str, name) for fold in range(n_folds)]
        self.summary_logger = Logger(name=f'{name}_summary')
        self._training_start: Optional[datetime] = None

    def write_config(self, config: dict) -> None:
        """

        """
        for logger in self.fold_logger + [self.summary_logger]:
            logger.write_config(config)

    def write_text(self, tag: str, text: str) -> None:
        """
        Writes a custom text to a custom tag in the TensorBoard log file.

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        for logger in self.fold_logger + [self.summary_logger]:
            logger.write_text(tag, text)

    def write_model(self, model: nn.Module) -> None:
        """
        Writes the model architecture formatted as text to the TensorBoard log file.

        Args:
            model (nn.Module): The model.
        """
        for logger in self.fold_logger + [self.summary_logger]:
            logger.write_model(model)

    def summarize(self) -> None:
        """
        Summarizes the mean training loss for all folds.

        This method calculates the mean training loss, test loss, and accuracy for each epoch
        that is saved in a list of loggers and writes the mean values to the TensorBoard log file.
        :param loggers: A list of loggers for each fold of a cross validation.
        """

        for key, value in self.fold_logger[0].text.items():
            self.summary_logger.write_text(key, value)

        if hasattr(self.fold_logger[0], 'model'):
            self.summary_logger.write_model(self.fold_logger[0].model)

        for row in self.fold_logger[0].train_loss.keys():
            mean_train_loss = np.mean([fold.train_loss[row] for fold in self.fold_logger])
            self.summary_logger.log_loss(mean_train_loss, row, '1_train')

            if row in self.fold_logger[0].val_loss.keys():
                mean_val_loss = np.mean([fold.val_loss[row] for fold in self.fold_logger])
                self.summary_logger.log_loss(mean_val_loss, row, '2_validation')

            if row in self.fold_logger[0].test_loss.keys():
                mean_test_loss = np.mean([fold.test_loss[row] for fold in self.fold_logger])
                self.summary_logger.log_loss(mean_test_loss, row, '3_test')

            for score in self.fold_logger[0].scores.keys():
                for class_label in self.fold_logger[0].scores[score].keys():
                    if row in self.fold_logger[0].scores[score][class_label].keys():
                        mean_value = np.mean([fold.scores[score][class_label][row] for fold in self.fold_logger])
                        self.summary_logger.log_test_score(mean_value, row, class_label, score)

    def get_final_scores(self):
        """
        Returns the final scores from the last epoch of the model for all folds as their mean a standard deviation.

        Iterates over all scores and all classes and all folds. Writes for each of these combindations the last value
        of the score into a dictionary and calculate for each of these combinations the mean and the standard
        deviation for all folds and write this into the dict as well.
        """
        final_scores = {}
        for score in self.fold_logger[0].scores.keys():
            for class_label in self.fold_logger[0].scores[score].keys():
                for i, fold in enumerate(self.fold_logger):
                    final_scores[f'{score}_{class_label}_fold{i}'] = fold.scores[score][class_label][
                        max(fold.scores[score][class_label].keys())]

                mean = np.mean([fold.scores[score][class_label][max(fold.scores[score][class_label].keys())] for fold in
                                self.fold_logger])
                std = np.std([fold.scores[score][class_label][max(fold.scores[score][class_label].keys())] for fold in
                              self.fold_logger])
                final_scores[f'{score}_{class_label}_mean'] = mean
                final_scores[f'{score}_{class_label}_std'] = std

        return final_scores

    def close(self):
        self.summary_logger.write_training_end("Summary finished")


class MultiInstanceLogger:
    """
    A class used to log training information for multiple instances of a model.
    """

    def __init__(self, n_folds: int, name: str = "") -> None:

        self.name = name
        self.n_folds = n_folds
        self.logger = None
        self.start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.results = []
        self.text = {}

    def next_logger(self) -> CrossValLogger:
        """
        Initializes a new crossvalidation logger.

        Returns: CrossValLogger
        """
        if self.n_folds == 0:
            self.logger = Logger(self.name)
        else:
            self.logger = CrossValLogger(n_folds=self.n_folds, name=self.name)

        # Write saved properties into new logger
        for key, value in self.text.items():
            self.logger.write_text(key, value)
        self.logger.write_config(self.config)

        return self.logger

    def collect_final_results(self, train_params: dict, model_params:dict) -> None:
        """
        Collects the final results of the active loggers.

        This method collects the final results from the current logger, adds train and model params from passed
        arguments and stores them into the list of results.

        Args:
            train_params (dict): The training parameters.
            model_params (dict): The model parameters.
        """

        results = self.logger.get_final_scores()
        results.update(train_params)
        results.update(model_params)
        results.update({'name': f'{self.logger.start_time_str}_{self.name}'})

        self.results.append(results)

    def save_final_results(self) -> None:
        """
        Saves the final results of the active loggers to a pandas DataFrame.

        This method saves the final results from the current logger into a pandas DataFrame.
        """
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(f'data/output/gs_results/{self.start_time_str}_{self.name}_results.csv', index=False)

    def write_config(self, config: dict) -> None:
        """
        Saves the config to writes it into each new logger instance

        Args:
            config (dict): The configuration of the training.
        """
        self.config = copy.deepcopy(config)

    def write_text(self, tag: str, text: str) -> None:
        """
        Saves the text to write it into each new logger instance

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        self.text[tag] = text

    def write_model(self, model: nn.Module) -> None:
        """
        Saves the model to write it into each new logger instance

        Args:
            model (nn.Module): The model.
        """
        self.model = model

    def close(self) -> None:
        """
        Closes multi instance logger and saves the results into a pandas DataFrame.
        """
        df_results = pd.DataFrame(self.results)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        df_results.to_csv(f'runs/{current_time}_{self.name}_results.csv', index=False)
