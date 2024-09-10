import copy
from datetime import datetime
from typing import Optional

from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.logger.logger import Logger


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
        self.name = f"runs/{current_time_string}_{name}_fold{fold}"
        self._summary_writer = SummaryWriter(self.name)
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

    def write_dict(self, data: dict, name: str="config") -> None:
        """
        Logs the configuration of the training.

        This method writes for each first level item in the config a text of its contents with the dict key as subtag.

        Args:
            data (dict): The configuration of the training.
        """
        data = copy.deepcopy(data)
        super().write_dict(data, name)
        for key, value in data.items():
            self.text[f'{name}_{key}'] = value

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
        super().log_model_path(model_path)
        self.text["model/mode_path"] = model_path
