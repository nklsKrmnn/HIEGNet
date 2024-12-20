from itertools import product
import random
import copy

from src.pipelines.cross_validation import cross_validation
from src.logger.multi_instance_logger import MultiInstanceLogger
from src.utils.constants import PARAMETER_SEARCH_SPACE


# Set random seed
random.seed(42)

def grid_search(model_name: str,
                model_attributes: dict,
                logger: MultiInstanceLogger,
                dataset,
                device,
                trainer_class,
                training_parameters: dict,
                n_folds: int = 5,
                start_index: int = 0
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
    :param start_index: Index to start the grid search at. Defaults to 0
    :return: None
    """
    config_params = {
        "model_parameters": model_attributes,
        "training_parameters": training_parameters
    }

    grid = SearchSampler(config_params, PARAMETER_SEARCH_SPACE)

    grid.create_grid_search_space()
    grid.shuffle()

    size = grid.get_size()

    results = []

    print(f"Start grid search with {size} settings")

    for i in range(start_index, size):
        print("###################################")
        print(f"Setting {i + 1}/{size}")
        print("###################################")
        setting = grid[i]
        model_attributes = setting["model_parameters"]
        training_parameters = setting["training_parameters"]
        temp_logger = logger.next_logger()

        training_parameters['reported_set'] = 'val'  # Always report on validation set for grid search

        temp_logger.write_text("config_model_parameters", str(model_attributes))
        temp_logger.write_text("config_training_parameters", str(training_parameters))

        cross_validation(model_name=model_name,
                         model_attributes=model_attributes,
                         logger=temp_logger,
                         dataset=dataset,
                         device=device,
                         trainer_class=trainer_class,
                         training_parameters=training_parameters,
                         n_folds=n_folds)

        logger.collect_final_results(train_params=training_parameters, model_params=model_attributes)
        logger.save_final_results()


class SearchSampler:

    def __init__(self,
                 config: dict,
                 parameter_search_space: dict
                 ):
        self.config = config
        self.parameter_search_space = parameter_search_space

        # Recursive init for sub parameters
        for key, value in self.config.items():
            if isinstance(value, dict):
                self.config[key] = SearchSampler(value, parameter_search_space)

        self.search_space = []

    def create_grid_search_space(self) -> list:
        """
        Creates a grid search space over the model and training parameters that are marked with "gs" in the dictionaries.
        All other parameters are kept constant.
        """
        # Get all parameters that should be optimized for the model and training (where value is 'gs')
        search_parameters = [k for k, v in self.config.items() if v == 'gs']

        # Recursive call of the function for sub parameters
        for key, value in self.config.items():
            if isinstance(value, SearchSampler):
                self.parameter_search_space.update({key: value.create_grid_search_space()})
                search_parameters.append(key)

        # Create a list of all possible combinations of the search space
        for params in product(*[self.parameter_search_space[k] for k in search_parameters]):
            config = self.config.copy()

            for i, key in enumerate(search_parameters):
                config[key] = params[i]

            self.search_space.append(copy.deepcopy(config))

        return self.search_space

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
