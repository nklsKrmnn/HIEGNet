from typing import Final
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.wsi_preprocessing.functions.hand_crafted_feature import extract_shape_based_features, extract_texture_based_features
from src.wsi_preprocessing.functions.image_io import get_paths, isolate_central_mask
from src.wsi_preprocessing.functions.path_io import extract_patient_id, extract_index

PROJECT: Final[str] = "EXC"

input_dir = '/home/dascim/data/2_images_preprocessed/EXC/patches_glom_isolated/25/'
output_dir = f"/home/dascim/data/3_extracted_features/{PROJECT}/"

PATIENTS = ['006', '005']

input_mask_paths = get_paths(input_dir, ".png")

list_image_features = []

for path in tqdm(input_mask_paths):
    patient = extract_patient_id(path)

    if patient not in PATIENTS:
        continue

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Isolate the central glomerulus
    isolated_glom_image = isolate_central_mask(image)

    # Binarize the isolated glomerulus
    isolated_glom_bin_image = (isolated_glom_image > 0).astype(np.uint8)

    # Extract features
    features = extract_shape_based_features(isolated_glom_bin_image)
    features.update(extract_texture_based_features(isolated_glom_image))

    features['patient'] = extract_patient_id(path)
    features['glom_index'] = extract_index(path)

    list_image_features.append(features)

df_features = pd.DataFrame(list_image_features)

df_features.to_csv(f"{output_dir}/glom_features_uniform_56.csv", index=False)



