from typing import Final
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.wsi_preprocessing.functions.hand_crafted_feature import extract_shape_based_features, \
    extract_texture_based_features
from src.wsi_preprocessing.functions.image_io import get_paths, isolate_central_contour, get_subfolders, binning_image
from src.wsi_preprocessing.functions.path_io import extract_patient_id, extract_index

PROJECT: Final[str] = "EXC"

input_dir = '/home/dascim/data/2_images_preprocessed/EXC/augmentations/'
mask_dir = '/home/dascim/data/2_images_preprocessed/EXC/masks_glom_isolated/25'
output_dir = f"/home/dascim/data/3_extracted_features/{PROJECT}/"

PATIENTS = ['006', '005', '004']

input_sub_dirs = get_subfolders(input_dir)

for aug_dir in input_sub_dirs:

    print(f"Processing {aug_dir}")

    augmentation_name = aug_dir.split("/")[-1]

    input_patch_paths = get_paths(aug_dir, ".png")
    input_patch_paths = [path for path in input_patch_paths if extract_patient_id(path) in PATIENTS]

    list_image_features = []

    for path in tqdm(input_patch_paths):
        central_glom_mask = cv2.imread(f"{mask_dir}/patch_p{path.split('_p')[-1]}", cv2.IMREAD_GRAYSCALE)

        if central_glom_mask is None:
            print(f"Mask not found for {path}")
            continue

        central_glom_mask = binning_image(central_glom_mask, 2)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Fill cut off with zeros
        image_size = image.shape[0]
        if central_glom_mask.shape[0] < image_size or central_glom_mask.shape[1] < image_size:
            central_glom_mask = np.pad(central_glom_mask, (
                (0, image_size - central_glom_mask.shape[0]), (0, image_size - central_glom_mask.shape[1])), 'constant')


        # Isolate the central glomerulus
        isolated_glom_image = image * central_glom_mask

        # Binarize the isolated glomerulus
        isolated_glom_bin_image = (isolated_glom_image > 0).astype(np.uint8)

        # Extract features
        features = extract_shape_based_features(isolated_glom_bin_image)
        features.update(extract_texture_based_features(isolated_glom_image))

        features['patient'] = extract_patient_id(path)
        features['glom_index'] = extract_index(path)

        list_image_features.append(features)

    df_features = pd.DataFrame(list_image_features)

    df_features.to_csv(f"{output_dir}/augmentations/glom_features_{augmentation_name}.csv", index=False)
