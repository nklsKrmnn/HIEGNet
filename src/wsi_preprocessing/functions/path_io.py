import os
import re
from typing import Union

import numpy as np


def extract_patient_id(file_path: str) -> str:
    return re.search(r"_p(\d+)_", file_path).group(1)

def extract_staining_id(file_path: str) -> str:
    return re.search(r"_s(\d+)_", file_path).group(1)

def extract_index(file_path: str) -> str:
    return re.search(r"_i(\d+)\.", file_path).group(1)


def get_path_up_to(current_path: str, directory_name: str):
    """
    Results the path up to the specified directory name. Current path can be passed with os.getcwd().

    :param current_path: String of current path.
    :param directory_name: String of directory name to stop at.
    :return: String of path up to the specified directory name.
    """


    # Split the path into its components
    path_components = current_path.split(os.sep)

    # Find the index of the target directory in the path components
    if directory_name in path_components:
        target_index = path_components.index(directory_name)
        # Join the path components up to and including the target directory
        truncated_path = os.sep.join(path_components[:target_index])
        return truncated_path
    else:
        raise ValueError(f"Directory '{directory_name}' not found in the current file path")

def create_image_path(patient: str, stain: str, index: str, image_dir: str):
    if int(index) > -1:
        return f"{image_dir}/{stain}/patch_p{patient}_s{stain}_i{index}.png"
    else:
        return np.nan