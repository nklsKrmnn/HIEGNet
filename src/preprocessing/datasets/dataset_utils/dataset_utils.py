import os
import numpy as np
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

def get_train_val_test_indices(y, test_split, val_split, random_seed, test_patients, val_patients):
    if (test_split < 1.0 and test_split > 0.0):
        train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=float(test_split),
                                                       random_state=random_seed, stratify=y.numpy())
        val_split_correction = test_split * val_split
    else:
        train_indices = np.arange(len(y))
        test_indices = np.arange(len(y)) if test_patients else np.array([])
        val_split_correction = 0

    if (val_split < 1.0 and val_split > 0.0):
        train_indices, val_indices = train_test_split(train_indices,
                                                      test_size=float(val_split + val_split_correction),
                                                      random_state=random_seed, stratify=y[train_indices].numpy())
    else:
        train_indices = np.arange(len(y))
        val_indices = np.arange(len(y)) if val_patients else np.array([])

    return train_indices, val_indices, test_indices