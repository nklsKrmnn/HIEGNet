"""
This module contains the Logger class which is used to log training information.
"""
import io
import os
from datetime import datetime
from typing import Optional

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
        self.name = f"runs/{current_time_string}_{name}"
        self._summary_writer = SummaryWriter(self.name)
        self._training_start: Optional[datetime] = None

    def write_text(self, tag: str, text: str) -> None:
        """
        Writes a custom text to a custom tag in the TensorBoard log file.

        Args:
            tag (str): The tag for the text.
            text (str): The text to write.
        """
        self._summary_writer.add_text(tag, str(text))

    def write_dict(self, data: dict, name:str='config') -> None:
        """
        Logs the configuration of the training.

        This method writes for each first level item in the config a text of its contents with the dict key as subtag.

        Args:
            data (dict): The configuration of the training.
        """
        for key, value in data.items():
            self.write_text(f'{name}_{key}', str(value))

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

    def save_figure(self, figure: plt.Figure, tag: str, index: int) -> None:
        """
        Saves an image into logger with tag and at index.

        :param figure: Figure to safe
        :param tag: Tag in image folder
        :param index: Index of image
        :return: None
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        self._summary_writer.add_image(f"{tag}", image, index)

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


