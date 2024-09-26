import os
from typing import Union

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def list_annotation_file_names(dir_path: str) -> list:
    """
    List all annotation file names (if it ends with ".csv") in the given directory.

    Args:
        dir_path (str): The directory path.

    Returns:
        list: List of file names.
    """
    file_names = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                file_names.append(os.path.join(root, file))
    return file_names


def list_neighborhood_image_paths(patient: str, dir_path: str) -> list:
    """
    Load the neighborhood images for a patient.

    The neighborhood images are loaded from the dir_path directory and the images.

    :param patient: The patient id
    :return: List of paths of the neighborhood images for one graph
    """
    if int(patient) >= 10:
        raise ValueError("Implement this correctly for patients >= 10.")

    image_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.png') and f"p00{patient}" in file:
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_train_val_test_indices(y: Union[torch.tensor, pd.Series, np.array, list],
                               test_split,
                               val_split,
                               random_seed,
                               is_test_patient,
                               is_val_patient,
                               glom_indices: list,
                               set_indices_path: str,
                               split_action: str = "load") -> tuple[list, list, list]:
    """
    Get the train, validation and test indices.

    Generates the train, validation and test indices for the given y values. The indices are either loaded from a
    file (split_action = "load") or generated based on the given parameters (split_action = "split"). The indices are
    saved to a file if split_action is 'save'.

    Args:
        y (Union[torch.tensor, pd.Series, np.array, list]): Target values.
        test_split (float): Test split ratio.
        val_split (float): Validation split ratio.
        random_seed (int): Random seed.
        is_test_patient (bool): If the patient is used in the test set.
        is_val_patient (bool): If the patient is used in the validation set.
        glom_indices (list): List of glomeruli indices.
        set_indices_path (str): Path where to save the files.
        split_action (str): Action to take, either 'load', 'split' or 'save'.

    Returns:
        tuple: Tuple of lists with train, val and test indices

    """

    if split_action == "load":
        # Load indices from files
        train_indices, val_indices, test_indices = load_indices(glom_indices, set_indices_path)

        # Change indices to different set based on given parameters
        if not is_test_patient or test_split == 0:
            # Move test indices to train set if this patient is not in the test set
            train_indices = train_indices + test_indices
            test_indices = []
        if not is_val_patient or val_split == 0:
            # Move val indices to train set if this patient is not in the val set
            train_indices = train_indices + val_indices
            val_indices = []
        if test_split == 1 and is_test_patient:
            # Move all indices to test set if we test on the whole patient
            test_indices = test_indices + train_indices + val_indices
            train_indices = []
            val_indices = []

    else:
        if (test_split == 1) and (val_split == 1):
            raise ValueError("Both test and validation split cannot be 1.")

        # Transform y to numpy array
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        elif isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, list):
            y = np.array(y)

        # Split test indices from whole set
        if (test_split > 0.0) and (test_split < 1) and is_test_patient:
            train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=float(test_split),
                                                           random_state=random_seed, stratify=y)
            val_split_correction = test_split * val_split
        elif (test_split == 1.0) and is_test_patient:
            train_indices = np.array([])
            test_indices = np.arange(len(y))
            val_split_correction = 0
        else:
            val_split_correction = 0
            test_indices = np.array([])
            train_indices = np.arange(len(y))

        # Split remaining indices into train and val set
        if (val_split > 0.0) and is_val_patient:
            train_indices, val_indices = train_test_split(train_indices,
                                                          test_size=float(val_split + val_split_correction),
                                                          random_state=random_seed, stratify=y[train_indices])
        else:
            val_indices = np.array([])

        # Make lists from arrays
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()
        test_indices = test_indices.tolist()

        # Save indices to file if required
        if split_action == "save":
            save_indices(train_indices, val_indices, test_indices, glom_indices, set_indices_path)

    return train_indices, val_indices, test_indices


def create_mask(num_nodes, indices) -> torch.tensor:
    """
    Create a mask for the train, validation or test data.

    Creates a torch mask tensor with True values for the indices of the given indices list.

    Args:
        num_nodes (int): Number of nodes.
        indices (list): List of indices.

    Returns:
        torch.tensor: Mask tensor.

    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask


def save_indices(train_indices: list,
                 val_indices: list,
                 test_indices: list,
                 glom_indices: list,
                 set_indices_path: str):
    """
    Save the indices to a file.

    Saves glomeruli indices of the patient for the train, val and test set in separate files at set_indices_path. The
    indices for the different set are given as indices of the glom index list. Check for each glomeruli index first
    if in already exists in the file, to avoid duplicates.

    Args:
        train_indices (list): List of train indices.
        val_indices (list): List of validation indices.
        test_indices (list): List of test indices.
        glom_indices (list): List of glomeruli indices for the patient.
        set_indices_path (str): Path where to save the files.

    """
    for i, indices in enumerate([train_indices, val_indices, test_indices]):
        # Create file name
        file_path = f'{set_indices_path}_{["train", "val", "test"][i]}.txt'

        # Read existing indices
        ecisting_indices = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                ecisting_indices = f.read().splitlines()
                ecisting_indices = [int(index) for index in ecisting_indices]

        # Write indices into file
        with open(file_path, 'a') as f:
            for index in indices:
                if glom_indices[index] not in ecisting_indices:
                    f.write(f'{glom_indices[index]}\n')


def load_indices(glom_indices: list,
                 set_indices_path: str) -> tuple[list, list, list]:
    """
    Loads set indices specific to the parameters.

    Loads the indices from a file a set_indices_path, that is named with the given parameters val_split, test_split
    and random_seed. It returns the indices of the glom_indices in the list, where the glom_index can be found in the
    loaded list of indices specific to a set.

    Args:
        glom_indices (list): List of glomeruli indices.
        set_indices_path (str): Path where the files are saved.

    Returns:
        tuple: Tuple of lists with train, val and test indices.
    """

    for i, set_type in enumerate(["train", "val", "test"]):
        # Get all glom indices from file for specific set type for all patients
        file_name = f'{set_indices_path}_{set_type}.txt'
        with open(file_name, 'r') as f:
            set_glom_indices = f.read().splitlines()
            set_glom_indices = [int(index) for index in set_glom_indices]

        # Write respective set indices into variables
        # See if the glom index exists in the glom indices for this set type
        set_indices = [i for i, glom_index in enumerate(glom_indices) if glom_index in set_glom_indices]
        if i == 0:
            train_indices = set_indices
        elif i == 1:
            val_indices = set_indices
        else:
            test_indices = set_indices

    return train_indices, val_indices, test_indices
