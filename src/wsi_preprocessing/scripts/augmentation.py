import os
from typing import Final
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from skimage.color import separate_stains, rbd_from_rgb, combine_stains, rgb_from_rbd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.wsi_preprocessing.functions.hand_crafted_feature import extract_shape_based_features, \
    extract_texture_based_features
from src.wsi_preprocessing.functions.image_io import get_paths, isolate_central_contour
from src.wsi_preprocessing.functions.path_io import extract_patient_id, extract_index
from wsi_preprocessing.functions.color_transformations import transform_to_florescent, transform_to_rgb


def shift_intensity(patch: np.array, shift: float) -> np.array:
    """
    Shift the intensity of a patch.

    Parameters
    ----------
    patch : np.array
        Patch.
    shift : int
        Shift.

    Returns
    -------
    np.array
        Patch with shifted intensity.
    """

    return np.clip(patch * shift, 0, 250)

def shift_stain(patch: np.array, shift: float, channel: int) -> np.array:
    stain_channels = separate_stains(patch, rbd_from_rgb)
    stain_channels[:, :, channel] = stain_channels[:, :, channel] * shift
    shiftes_patch = combine_stains(stain_channels, rgb_from_rbd)
    return shiftes_patch*255


def adjust_contrast_linear(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust the contrast of an image by scaling pixel values around the midpoint (128).
    :param image: Input image as a NumPy array (assumed to be in [0, 255] and uint8)
    :param factor: Contrast factor. factor > 1 increases contrast; factor < 1 decreases contrast.
    :return: Contrast-adjusted image as a NumPy array (uint8)
    """
    # Convert image to float for processing
    image_float = image.astype(np.float32)

    # Scale pixel values relative to the midpoint (128)
    adjusted = (image_float - 128) * factor + 128

    # Clip the values to maintain valid pixel range and convert back to uint8
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)


def save_augmented_patch(patch: np.array, output_dir: str, file_name: str, augmentation: str) -> None:
    """
    Save an augmented patch.

    Parameters
    ----------
    patch : np.array
        Patch.
    output_dir : str
        Output directory.
    file_name : str
        File name.
    augmentation : str
        Augmentation.
    """
    file_name = file_name[5:]
    patch_dir = os.path.join(output_dir, augmentation)
    os.makedirs(patch_dir, exist_ok=True)  # Ensure the directory exists
    patch_path = os.path.join(patch_dir, f"{augmentation}_{file_name}")
    cv2.imwrite(patch_path, patch)

PROJECT: Final[str] = "EXC"

input_dir = '/home/dascim/data/2_images_preprocessed/EXC/patches_low_resolution/25/'
output_dir = f"/home/dascim/data/2_images_preprocessed/{PROJECT}/augmentations/"

PATIENTS = ['005','006']

input_path_paths = get_paths(input_dir, ".png")

# filter by patients
input_path_paths = [path for path in input_path_paths if extract_patient_id(path) in PATIENTS]

list_image_features = []



for path in tqdm(input_path_paths):
    file_name = path.split("/")[-1]

    patch = cv2.imread(path)

    patch_intensity_shifted_plus = shift_intensity(patch, 1.5)
    save_augmented_patch(patch_intensity_shifted_plus, output_dir, file_name, "intensity_plus5")

    patch_intensity_shifted_minus = shift_intensity(patch, 0.5)
    save_augmented_patch(patch_intensity_shifted_minus, output_dir, file_name, "intensity_minus5")

    patch_tcell_shifted_plus = shift_stain(patch, 1.5, 2)
    save_augmented_patch(patch_tcell_shifted_plus, output_dir, file_name, "stain_tcell_plus5")

    patch_tcell_shifted_minus = shift_stain(patch, 0.5, 2)
    save_augmented_patch(patch_tcell_shifted_minus, output_dir, file_name, "stain_tcell_minus5")

    patch_macro_shifted_plus = shift_stain(patch, 1.5, 1)
    save_augmented_patch(patch_macro_shifted_plus, output_dir, file_name, "stain_macro_plus5")

    patch_macro_shifted_minus = shift_stain(patch, 0.5, 1)
    save_augmented_patch(patch_macro_shifted_minus, output_dir, file_name, "stain_macro_minus5")

    patch_tissue_shifted_plus = shift_stain(patch, 1.5, 0)
    save_augmented_patch(patch_tissue_shifted_plus, output_dir, file_name, "stain_3_plus5")

    patch_tissue_shifted_minus = shift_stain(patch, 0.5, 0)
    save_augmented_patch(patch_tissue_shifted_minus, output_dir, file_name, "stain_3_minus5")

    patch_contrast_plus = adjust_contrast_linear(patch, 1.5)
    save_augmented_patch(patch_contrast_plus, output_dir, file_name, "contrast_plus5")

    patch_contrast_minus = adjust_contrast_linear(patch, 0.5)
    save_augmented_patch(patch_contrast_minus, output_dir, file_name, "contrast_minus5")
