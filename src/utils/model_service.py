"""
This module contains the ModelService class which provides methods for creating, saving, and loading PyTorch trained_models.
"""
import os
from pathlib import Path
from typing import Final, Optional

import torch
import torch.nn as nn

from src.utils.constants import MODEL_NAME_MAPPING
from utils.path_io import get_path_up_to

MODEL_OUTPUT_PATH: Final[Path] = Path("data", "output", "trained_models")
ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "src")


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
                     model_attributes: dict,
                     fold: int = None) -> nn.Module:
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
        #TODO: Work with version
        model_attributes = model_attributes.copy()

        # Use models from dir if available
        if "model_dir" in model_attributes.keys():
            # Read model files in dir
            dir_path = ROOT_DIR + model_attributes.pop("model_dir")
            model_files = os.listdir(dir_path)
            model_files = [f for f in model_files if f.endswith(".pt")]#
            model_files.sort()
            model_file = model_files[fold]
            model_path = dir_path + model_file
        elif "model_path" in model_attributes.keys():
            model_path = model_attributes.pop("model_path")
            model_path = ROOT_DIR + model_path
        else:
            model_path = None

        # Decompose the msg_passing_types attribute if it is set the same for all edge groups
        if 'msg_passing_types' in model_attributes.keys():
            if isinstance(model_attributes['msg_passing_types'], str):
                edge_groups = ['glom_to_glom', 'cell_to_glom', 'cell_to_cell']
                model_attributes['msg_passing_types'] = {key: model_attributes['msg_passing_types'] for key in edge_groups}

        model = MODEL_NAME_MAPPING[model_name](**model_attributes)

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
    def save_model(cls, model: nn.Module, name: str, new_version:bool=False) -> str:
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
        global path

        version = cls.get_latest_version(name)

        if version is None:
            version = 0

        if new_version:
            version += 1

        path = Path(MODEL_OUTPUT_PATH, f'{name}_v{version}.pt')

        # Save the model state dict to the specified path
        torch.save(model.state_dict(), path)
        return str(path.absolute())

    @classmethod
    def load_model(self, path: str, model_name: str, model_attributes: dict) -> nn.Module:
        """
        Loads a PyTorch model from the specified file.

        The method loads the model from the specified file and sets the model to evaluation mode.

        Args:
            path (str): The path to the file containing the model.

        Returns:
            nn.Module: The loaded model.
        """

        state_dict = torch.load(path)
        model = MODEL_NAME_MAPPING[model_name](**model_attributes)
        model.load_state_dict(state_dict)
        model.eval()

        return model

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
