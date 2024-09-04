import copy
from datetime import datetime
from typing import Optional

import pandas as pd
from torch import nn as nn

from src.logger.cross_val_logger import ManyFoldLogger
from src.logger.logger import Logger


class MultiInstanceLogger:
    """
    A class used to log training information for multiple instances of a model.
    """
    logger: Optional[ManyFoldLogger]

    def __init__(self, n_folds: int, name: str = "", report_max_scores: bool = True) -> None:

        self.name = name
        self.n_folds = n_folds
        self.logger = None
        self.report_max_scores = report_max_scores # Whether to report scores with max value or last scores
        self.start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.results = []
        self.text = {}

    def next_logger(self) -> ManyFoldLogger:
        """
        Initializes a new crossvalidation logger.

        Returns: CrossValLogger
        """
        if self.n_folds == 0:
            self.logger = Logger(self.name)
        else:
            self.logger = ManyFoldLogger(n_folds=self.n_folds, name=self.name)

        # Write saved properties into new logger
        for key, value in self.text.items():
            self.logger.write_text(key, value)
        self.logger.write_dict(self.config)

        return self.logger

    def collect_final_results(self, train_params: dict, model_params:dict) -> None:
        """
        Collects the final results of the active loggers.

        This method collects the final results from the current logger, adds train and model params from passed
        arguments and stores them into the list of results. Depending on the class attribute report_max_scores, either
        the last scores or the scores of the model in the best epoch are collected.

        Args:
            train_params (dict): The training parameters.
            model_params (dict): The model parameters.
        """

        results = self.logger.get_max_scores() if self.report_max_scores else self.logger.get_final_scores()
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

    def write_dict(self, data: dict, name:str='config') -> None:
        """
        Saves the config to writes it into each new logger instance

        Args:
            data (dict): The configuration of the training.
        """
        if name == 'config':
            self.config = copy.deepcopy(data)
        elif name == 'score':
            self.score = copy.deepcopy(data)

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
        #TODO Remove function?
        """
        Closes multi instance logger and saves the results into a pandas DataFrame.
        """
        df_results = pd.DataFrame(self.results)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        #df_results.to_csv(f'runs/{current_time}_{self.name}_results.csv', index=False)
