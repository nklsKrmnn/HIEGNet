import pandas as pd
from itertools import product
import random
import copy

from src.utils.logger import Logger, CrossValLogger, MultiInstanceLogger
from src.utils.constants import MODEL_PARAMETER_SEARCH_SPACE, TRAIN_PARAMETER_SEARCH_SPACE
from src.utils.model_service import ModelService


def grid_search(model_name: str,
                model_attributes: dict,
                logger: MultiInstanceLogger,
                dataset,
                device,
                trainer_class,
                training_parameters: dict,
                n_folds: int = 5
                ):
    """
    Perform a grid search over the model and training parameters

    First creates a grid search space over all training parameters as model attributes that are marked wih "gs" in the
    dictionaries. Then, for each setting in the grid search space, performs cross validation and stores the results.
    Before interating over the grid search space, the grid is shuffled. The setting parameters as well as the results of
    the crossvalidation are saved into a large list of dictionaries which is later converted into a pandas DataFrame.

    :param model_name: Name of the model to use
    :param model_attributes: Dict of model attributes with 'gs' at all parameters that should be optimized
    :param logger: A multi instance logger to log the results
    :param dataset: Dataset to use
    :param device: Device to use
    :param training_parameters: Dict of training parameters with 'gs' at all parameters that should be optimized
    :param n_folds: Number of folds to use for crossvalidation
    :return:
    """
    grid = SearchSampler(model_attributes, training_parameters, MODEL_PARAMETER_SEARCH_SPACE,
                         TRAIN_PARAMETER_SEARCH_SPACE)

    grid.create_grid_search_space()
    grid.shuffle()

    size = grid.get_size()

    results = []

    print(f"Start grid search with {size}settings")

    for i in range(0, size):
        print("###################################")
        print(f"Setting {i + 1}/{size}")
        print("###################################")
        setting = grid[i]
        model_attributes = setting[0]
        training_parameters = setting[1]
        temp_logger = logger.next_logger()

        training_parameters['reported_set'] = 'val'  # Always report on validation set for grid search

        temp_logger.write_text("config_model_parameters", str(model_attributes))
        temp_logger.write_text("config_training_parameters", str(training_parameters))

        final_scores = cross_validation(model_name=model_name,
                                        model_attributes=model_attributes,
                                        logger=temp_logger,
                                        dataset=dataset,
                                        device=device,
                                        trainer_class=trainer_class,
                                        training_parameters=training_parameters,
                                        n_folds=n_folds)

        logger.collect_final_results(train_params=training_parameters, model_params=model_attributes)
        logger.save_final_results()
        results.append({**model_attributes, **training_parameters, **final_scores})

    results = pd.DataFrame(results)

    return results


def cross_validation(model_name: str,
                     model_attributes: dict,
                     logger: CrossValLogger,
                     dataset,
                     device,
                     trainer_class,
                     training_parameters: dict,
                     n_folds: int = 5
                     ) -> dict:
    """

    :param model_name:
    :param model_attributes:
    :param logger:
    :param dataset:
    :param device:
    :param training_parameters:
    :param n_folds:
    :return: A dictionary containing the final scores of the model for all folds as their mean a standard deviation
    """
    dataset.create_folds(n_folds)

    for fold in range(n_folds):
        model = ModelService.create_model(model_name=model_name,
                                          model_attributes=model_attributes)
        logger.fold_logger[fold].write_model(model)
        dataset.activate_fold(fold)

        trainer = trainer_class(
            dataset=dataset,
            model=model,
            device=device,
            logger=logger.fold_logger[fold],
            **training_parameters)

        trainer.start_training()
        trainer.save_model()

    final_scores = logger.get_final_scores()

    logger.summarize()
    logger.close()

    return final_scores


class SearchSampler:

    def __init__(self,
                 model_attributes: dict,
                 training_parameters: dict,
                 model_parameter_search_space: dict,
                 train_parameter_search_space: dict
                 ):
        self.model_attributes = model_attributes
        self.training_parameters = training_parameters
        self.model_parameter_search_space = model_parameter_search_space
        self.train_parameter_search_space = train_parameter_search_space
        self.search_space = []

    def create_grid_search_space(self) -> None:
        """
        Creates a grid search space over the model and training parameters that are marked with "gs" in the dictionaries.
        All other parameters are kept constant.
        """
        # Get all parameters that should be optimized for the model and training (where value is 'gs')
        search_training_parameters = {k: v for k, v in self.training_parameters.items() if v == 'gs'}
        search_model_attributes = {k: v for k, v in self.model_attributes.items() if v == 'gs'}

        base_epochs = self.training_parameters['epochs']

        # Create a list of all possible combinations of the search space
        # TODO implement function to get special search spaces
        for model_params in product(*[self.model_parameter_search_space[k] for k in search_model_attributes.keys()]):
            for train_params in product(
                    *[self.train_parameter_search_space[k] for k in search_training_parameters.keys()]):

                model_attributes = self.model_attributes.copy()
                training_parameters = self.training_parameters.copy()

                for i, key in enumerate(search_model_attributes.keys()):
                    model_attributes[key] = model_params[i]

                for i, key in enumerate(search_training_parameters.keys()):
                    training_parameters[key] = train_params[i]

                # Adjust max lr for scheduler
                if "lr_scheduler_params" in training_parameters.keys():
                    training_parameters["lr_scheduler_params"]['params']["max_lr"] = training_parameters[
                                                                                         "learning_rate"] * 10

                # Adjust epochs to learning rate
                if 'learning_rate' in search_training_parameters:
                    training_parameters["epochs"] = max(
                        int(0.0003 / training_parameters['learning_rate'] * base_epochs), 1)

                # Check that gcn is only used for homogenous edge types
                hetero_edge_types = [et for et in model_attributes['msg_passing_types'].keys() if
                                     et.split('_')[0] != et.split('_')[2]]
                if not any(model_attributes['msg_passing_types'][et] == 'gcn' for et in hetero_edge_types):
                    # Append the setting to the search space
                    self.search_space.append((copy.deepcopy(model_attributes), copy.deepcopy(training_parameters)))

    def shuffle(self) -> None:
        """
        Shuffles the search space
        """
        random.shuffle(self.search_space)

    def get_size(self) -> int:
        """
        Returns the size of the search space
        """
        return len(self.search_space)

    def __getitem__(self, item):
        return self.search_space[item]
