import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def list_annotation_file_names(dir_path:str) -> list:
    file_names = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                file_names.append(os.path.join(root, file))
    return file_names

def list_neighborhood_image_paths(patient:str, dir_path:str) -> list:
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


def get_train_val_test_indices(y, test_split, val_split, random_seed, is_test_patient, is_val_patient):
    if (test_split > 0.0) and is_test_patient:
        train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=float(test_split),
                                                       random_state=random_seed, stratify=y.numpy())
        val_split_correction = test_split * val_split
    else:
        val_split_correction = 0
        test_indices = []
        train_indices = np.arange(len(y))

    if (val_split > 0.0) and is_val_patient:
        train_indices, val_indices = train_test_split(train_indices,
                                                      test_size=float(val_split + val_split_correction),
                                                      random_state=random_seed, stratify=y[train_indices].numpy())
    else:
        val_indices = []

    return train_indices, val_indices, test_indices


def create_mask(num_nodes, indices) -> torch.tensor:
    """
    Create a mask for the train, validation or test data.

    Creates a torch mask tensor with True values for the indices of the given indices list.
    :param num_nodes: Number of nodes in mask
    :param indices: Indices to be True
    :return: Mask
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask
