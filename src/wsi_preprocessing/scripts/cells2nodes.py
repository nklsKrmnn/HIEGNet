import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm as tdqm

from src.wsi_preprocessing.functions.image_io import get_paths, upsampling_image
from src.wsi_preprocessing.functions.path_io import extract_index, extract_patient_id, get_path_up_to

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")

CELL_TYPE = "tcell"
stain = 25
input_dir = f"{ROOT_DIR}/data/3_extracted_features/EXC/masks_cellpose/{CELL_TYPE}"
annotations_dir = f"{ROOT_DIR}/data/1_cytomine_downloads/EXC/annotations/{stain}"
glom_mask_dir = f"{ROOT_DIR}/data/2_images_preprocessed/EXC/masks_glom_isolated/{stain}"
output_dir = f"{ROOT_DIR}/data/3_extracted_features/EXC/cell_nodes/"

cell_masks_file_paths = get_paths(input_dir, ".npy")

cells = []
for i, cell_masks_path in tdqm(enumerate(cell_masks_file_paths), total=len(cell_masks_file_paths)):
    print(f"Processing ({1 + i}/{len(cell_masks_file_paths)}): {cell_masks_path}")
    # Load cell mask
    mask = np.load(cell_masks_path)

    # Extract information
    glom_index = extract_index(cell_masks_path)
    patient = extract_patient_id(cell_masks_path)
    n_cells = mask.max()

    # Load annotations
    annotation_path = f"{annotations_dir}/annotations_{patient}_{stain}.csv"
    annotations = pd.read_csv(annotation_path)

    glom_center = annotations[annotations["ID"] == int(glom_index)][["Center X", "Center Y"]].values[0]

    # Load center glom mask
    glom_mask = cv2.imread(f"{glom_mask_dir}/patch_p{patient}_s{stain}_i{glom_index}.png", cv2.IMREAD_GRAYSCALE)
    glom_mask = upsampling_image(glom_mask, mask.shape[0] // glom_mask.shape[0], interpolation=cv2.INTER_NEAREST)

    for j in range(1, n_cells):
        cell_mask = mask == j
        cell_mask = cell_mask.astype(np.uint8)

        # Determine, if cell is inside glomerulus
        is_in_glom = np.any(cell_mask & glom_mask)
        test = np.sum(cell_mask)
        # Calculate cell properties
        props = regionprops(cell_mask)
        center = props[0].centroid
        area = props[0].area
        eccentricity = props[0].eccentricity
        aspect_ratio = props[0].major_axis_length / props[0].minor_axis_length
        circularity = 4 * np.pi * area / props[0].perimeter ** 2
        perimeter = props[0].perimeter
        solidity = props[0].solidity

        # safe mask
        plt.imsave("isolated_macro.png", cell_mask, cmap='gray')

        center_global = (glom_center[0] - cell_mask.shape[1] // 2 + center[1],
                         glom_center[1] + cell_mask.shape[1] // 2 - center[0])

        cells.append({"center_x_local": center[1],
                      "center_y_local": center[0],
                      "center_x_global": center_global[0],
                      "center_y_global": center_global[1],
                      "area": area,
                      "eccentricity": eccentricity,
                      "aspect_ratio": aspect_ratio,
                      "circularity": circularity,
                      "perimeter": perimeter,
                      "solidity": solidity,
                      "associated_glom": glom_index,
                      "is_in_glom": is_in_glom,
                      "patient": patient})

df_cells = pd.DataFrame(cells)
df_cells.to_csv(f"{output_dir}/{CELL_TYPE}_cell_nodes_prejoin.csv", index=False)
