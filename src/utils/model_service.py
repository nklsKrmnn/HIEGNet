"""
This module contains the ModelService class which provides methods for creating, saving, and loading PyTorch trained_models.
"""

from pathlib import Path
from typing import Final, Optional

import torch
import torch.nn as nn

from src.utils.constants import MODEL_NAME_MAPPING

MODEL_OUTPUT_PATH: Final[Path] = Path("data", "output", "trained_models")

first_save = True
path: str = ""


class ModelService():
    """
    A service class for creating, saving, and loading PyTorch trained_models.

    This class provides methods for creating a new model from a configuration, saving a
    model to a file, loading the latest version of a model from a file, and getting the
    latest version number of a model. The trained_models are saved in the `MODEL_OUTPUT_PATH`
    directory and have the extension '.pt'.
    """

    @classmethod
    def create_model(cls,
                     model_name: str,
                     model_attributes: dict) -> nn.Module:
        """
        Creates a new model from a configuration.

        This method creates a new model of the specified type with the specified attributes.

        Args:
            model_name (str): The name of the model to create.
            model_attributes (dict): The attributes to use when creating the model.

        Raises:
            KeyError: If the specified model name does not exist in the `MODEL_NAME_MAPPING`.
            TypeError: If the creation of the model fails due to an error with the model attributes.

        Returns:
            nn.Module: The created model.
        """
        model_path = model_attributes.pop("model_path") if "model_path" in model_attributes else None

        try:
            model = MODEL_NAME_MAPPING[model_name](**model_attributes)
        except KeyError as parse_error:
            raise (
                KeyError(f"The model '{model_name}' does not exist!")
            ) from parse_error
        except TypeError as model_error:
            raise (
                TypeError(
                    f"The creation of the {model_name} model failed with the following error message: {model_error}."
                )
            ) from model_error

        if model_path is not None:
            model.load_state_dict(torch.load(model_path))

        return model

    @classmethod
    def get_latest_version(cls, name: str) -> Optional[int]:
        """
        Returns the highest version number of a model with the specified name.

        This method searches the `MODEL_OUTPUT_PATH` directory for files that start
        with the specified name followed by '_v'. It extracts the version numbers from
        these file names and returns the highest one. If no such files are found, it returns 0.

        Args:
            name (str): The name of the model.

        Returns:
            int: The highest version number of the specified model, or 0 if no model is found.
        """
        # create output directory if it does not exist
        if not MODEL_OUTPUT_PATH.exists():
            MODEL_OUTPUT_PATH.mkdir(parents=True)

        relevant_file_names = list(map(lambda f: f.name, filter(lambda f: f.name.startswith(
            f'{name}_v'), MODEL_OUTPUT_PATH.iterdir())))

        # Split off the substrings before the v and after the . in the file
        # name
        version_numbers = list(
            map(lambda f: int(f.split('v')[-1].split('.')[0]), relevant_file_names))

        if version_numbers:
            return max(version_numbers)

        return None

    @classmethod
    def save_model(cls, model: nn.Module) -> str:
        """
        Saves the specified PyTorch model to a file in the `MODEL_OUTPUT_PATH` directory.

        The method first gets the class name of the model and the latest version number
        of this model class. If no previous versions are found, it sets the version number to 1.

        The model is then saved to a file with a name in the format '{model_class_name}_v{version}.pt'.
        The absolute path to the saved model file is returned.

        Args:
            model (nn.Module): The PyTorch model to be saved.

        Returns:
            str: The absolute path to the saved model file.
        """
        # Use the global variable to determine if this is the first run of the
        # program.
        global first_save, path
        # Get the class name of the model as string
        model_class_name = type(model).__name__
        version = cls.get_latest_version(model_class_name)

        if version is None:
            version = 0

        if first_save:
            path = Path(MODEL_OUTPUT_PATH,
                        f'{model_class_name}_v{version + 1}.pt')
            first_save = False

        # Save the model state dict to the specified path
        torch.save(model.state_dict(), path)
        return str(path.absolute())

    @classmethod
    def load_newest_model(cls, model_name: str) -> nn.Module:
        """
        Loads the latest version of a PyTorch model of the specified class from the
        `MODEL_OUTPUT_PATH` directory.

        This method first gets the latest version number of the model class.
        If no versions are found, it raises a ValueError. It then loads the model from
        the file with the highest version number. The loaded model is set to evaluation mode.

        Args:
            model_name (str): The name of the model to be loaded.

        Raises:
            ValueError: If no model is found in the `MODEL_OUTPUT_PATH` directory.

        Returns:
            nn.Module: The loaded model.
        """
        version = cls.get_latest_version(model.__name__)
        if version == 0:
            raise ValueError(
                f'No model of class {model.__name__} found in {MODEL_OUTPUT_PATH}')

        # Load model from torch state dict
        state_dict, params = torch.load(
            Path(MODEL_OUTPUT_PATH, f'{model_name}_v{version}.pt'))
        model = MODEL_NAME_MAPPING[model_name](**params)
        model.load_state_dict(state_dict)
        model.eval()

        return model
