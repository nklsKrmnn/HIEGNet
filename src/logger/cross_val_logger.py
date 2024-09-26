from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from torch import nn as nn

from src.logger.logger import Logger
from src.logger.fold_logger import FoldLogger


class ManyFoldLogger():
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
        self.name = name

    def write_dict(self, data: dict, name:str="config") -> None:
        """

        """
        for logger in self.fold_logger + [self.summary_logger]:
            logger.write_dict(data, name)

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
        # Get a set of all keys for the epochs for all logger
        all_epochs = set()
        for fold in self.fold_logger:
            all_epochs.update(fold.train_loss.keys())

        for key, value in self.fold_logger[0].text.items():
            if 'score' in key:
                mean = np.mean([fold.text[key] for fold in self.fold_logger])
                self.summary_logger.write_text(key, mean)
            else:
                self.summary_logger.write_text(key, value)

        if hasattr(self.fold_logger[0], 'model'):
            self.summary_logger.write_model(self.fold_logger[0].model)

        for epoch in all_epochs:
            losses = [fold.train_loss.get(epoch) for fold in self.fold_logger if epoch in fold.train_loss]
            mean_train_loss = np.mean(losses) if losses else 0
            self.summary_logger.log_loss(mean_train_loss, epoch, '1_train')

            if epoch in self.fold_logger[0].val_loss.keys():
                losses = [fold.val_loss.get(epoch) for fold in self.fold_logger if epoch in fold.val_loss]
                mean_val_loss = np.mean(losses) if losses else 0
                self.summary_logger.log_loss(mean_val_loss, epoch, '2_validation')

            if epoch in self.fold_logger[0].test_loss.keys():
                losses = [fold.test_loss.get(epoch) for fold in self.fold_logger if epoch in fold.test_loss]
                mean_test_loss = np.mean(losses) if losses else 0
                self.summary_logger.log_loss(mean_test_loss, epoch, '3_test')

            for score in self.fold_logger[0].scores.keys():
                for class_label in self.fold_logger[0].scores[score].keys():
                    if epoch in self.fold_logger[0].scores[score][class_label].keys():
                        scores = [fold.scores[score][class_label].get(epoch) for fold in self.fold_logger if epoch in fold.scores[score][class_label]]
                        mean_score = np.mean(scores) if scores else 0
                        self.summary_logger.log_test_score(mean_score, epoch, class_label, score)

    def get_final_scores(self):
        """
        Returns the final scores from the last epoch of the model for all folds as their mean and standard deviation.

        Iterates over all scores and all classes and all folds. Writes for each of these combindations the last value
        of the score into a dictionary and calculate for each of these combinations the mean and the standard
        deviation for all folds and write this into the dict as well.
        """
        final_scores = {}
        for score in self.fold_logger[0].scores.keys():
            for class_label in self.fold_logger[0].scores[score].keys():
                last_key = max(self.fold_logger[0].scores[score][class_label].keys())
                for i, fold in enumerate(self.fold_logger):
                    final_scores[f'{score}_{class_label}_fold{i}'] = fold.scores[score][class_label][last_key]

                mean = np.mean([fold.scores[score][class_label][last_key] for fold in self.fold_logger])
                std = np.std([fold.scores[score][class_label][last_key] for fold in self.fold_logger])
                final_scores[f'{score}_{class_label}_mean'] = mean
                final_scores[f'{score}_{class_label}_std'] = std

        # Add loss values
        for loss in ['train', 'val']:
            last_key = max(self.fold_logger[0].train_loss.keys())
            for i, fold in enumerate(self.fold_logger):
                final_scores[f'{loss}_loss_fold{i}'] = fold.train_loss[last_key] if loss == 'train' else fold.val_loss[last_key]

            mean = np.mean([final_scores[f'{loss}_loss_fold{i}'] for i in range(len(self.fold_logger))])
            std = np.std([final_scores[f'{loss}_loss_fold{i}'] for i in range(len(self.fold_logger))])

            final_scores[f'{loss}_loss_mean'] = mean
            final_scores[f'{loss}_loss_std'] = std

        return final_scores

    def get_max_scores(self, ranking_score: str="f1_macro") -> dict:
        """
        Returns the scores for the best performaing model from all epochs of the training for all folds as their mean
        and standard deviation.

        Determines the epoch with the best performing model with regard to the given score or loss for each fold.
        Iterates then over all scores and loss to write the values into a dict for this best performing model.

        :return: Dict with score and fold as key and max value as value
        """
        max_scores = {}
        for i, fold in enumerate(self.fold_logger):
            # Determine best epoch
            if ranking_score == 'val_loss':
                best_epoch = min(fold.val_loss, key=fold.val_loss.get)
            elif ranking_score == 'train_loss':
                best_epoch = min(fold.train_loss, key=fold.train_loss.get)
            elif ranking_score == 'test_loss':
                best_epoch = min(fold.test_loss, key=fold.test_loss.get)
            elif ranking_score in fold.scores.keys():
                best_epoch = max(fold.scores[ranking_score]['0_total'], key=fold.scores[ranking_score]['0_total'].get)
            else:
                raise ValueError(f"Ranking Score {ranking_score} is not in fold.scores")

            # Collection scores
            for score in fold.scores.keys():
                for class_label in fold.scores[score].keys():
                    max_scores[f'{score}_{class_label}_fold{i}'] = fold.scores[score][class_label][best_epoch]

            # Add loss values
            for loss in ['train', 'val']:
                max_scores[f'{loss}_loss_fold{i}'] = fold.train_loss[best_epoch] if loss == 'train' else fold.val_loss[
                    best_epoch]

        # Calculate means and std
        for score in self.fold_logger[0].scores.keys():
            for class_label in self.fold_logger[0].scores[score].keys():
                mean = np.mean([max_scores[f'{score}_{class_label}_fold{fold}'] for fold in range(len(self.fold_logger))])
                std = np.std([max_scores[f'{score}_{class_label}_fold{fold}'] for fold in range(len(self.fold_logger))])

                max_scores[f'{score}_{class_label}_mean'] = mean
                max_scores[f'{score}_{class_label}_std'] = std

        for loss in ['train', 'val']:
            mean = np.mean([max_scores[f'{loss}_loss_fold{i}'] for i in range(len(self.fold_logger))])
            std = np.std([max_scores[f'{loss}_loss_fold{i}'] for i in range(len(self.fold_logger))])

            max_scores[f'{loss}_loss_mean'] = mean
            max_scores[f'{loss}_loss_std'] = std

        return max_scores

    def save_test_scores(self, eval_csv_path: str) -> None:
        """
        Saves test scores from all fold logger and stores it in the csv.

        Retrieves test scores from each sub/fold logger, which are a train and test with a random weight
        initialisation of the model. Calculates the mean and the std for each score across all initialisations. Saves
        the aggretation as well as the separate scores together with the parameters as a new row into the csv,
        which is at the give path.

        :param eval_csv_path: Path of a csv, where test results will be saves
        :return: Nona
        """
        row = {}
        row.update({'name': self.name})

        score_keys = [key for key in self.fold_logger[0].text.keys() if 'score' in key]

        for k in score_keys:
            # Add scores for each fold
            row.update({f'{k}_init{i}': fold.text[k] for i, fold in enumerate(self.fold_logger)})

            # Calucate mean and std for each score
            mean = np.mean([fold.text[k] for fold in self.fold_logger])
            std = np.std([fold.text[k] for fold in self.fold_logger])
            row[k + '_mean'] = mean
            row[k + '_std'] = std

        row.update({f'dataset_param_{key}': value for key, value in self.fold_logger[0].text['config_dataset_parameters'].items()})
        row.update({f'training_param_{key}': value for key, value in self.fold_logger[0].text['config_training_parameters'].items()})
        row.update({f'model_param_{key}': value for key, value in next(iter(self.fold_logger[0].text['config_model_parameters'].values())).items()})

        try:
            scores_file = pd.read_csv(eval_csv_path)
        except:
            scores_file = pd.DataFrame({})

        row = pd.DataFrame([row])
        scores_file = pd.concat([scores_file, row], ignore_index=True)
        scores_file.to_csv(eval_csv_path, index=False)


    def close(self):
        self.summary_logger.write_training_end("Summary finished")
