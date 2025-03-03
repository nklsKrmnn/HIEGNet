import cv2
import pandas as pd
import numpy as np
import gc

import matplotlib.pyplot as plt
from src.wsi_preprocessing.functions.coordinate_transformer import CoordinateTransformater
from src.wsi_preprocessing.functions.image_io import crop_patch_png, binning_image, get_paths, isolate_central_mask
import os
from src.wsi_preprocessing.functions.path_io import get_path_up_to, extract_patient_id

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")

CLASSES = ['Healthy','Dead','Sclerotic']
STAINS = [25]
PATIENTS = ['005', "006"]

for stain in STAINS:
    # Get dir for masks
    mask_dir = f'{ROOT_DIR}/data/1_cytomine_downloads/EXC/roi/{stain}/masks/glomeruli/'

    # Get all files in the directory
    mask_paths = get_paths(mask_dir, ".png")
    output_dir = f"{ROOT_DIR}/data/2_images_preprocessed/EXC/masks_glom_isolated/{stain}"

    for mask_path in mask_paths:
        patient = mask_path.split("/")[-1].split("_")[2]
        mask = plt.imread(mask_path)

        if patient in PATIENTS:
            # Load mask
            #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Import annotations
            df_annotation = pd.read_csv(f'{ROOT_DIR}/data/1_cytomine_downloads/EXC/annotations/{stain}/annotations_{patient}_{stain}.csv')
            df_annotation = df_annotation[df_annotation['Term'].apply(lambda t: t in CLASSES)].reset_index(drop=True)

            # Initialize coordinate transformer
            magnification = 0.5
            rotation = 0
            mirror_y = True
            target_y_offset = mask.shape[0]

            transformer = CoordinateTransformater(magnification=magnification,
                                                  rotation=rotation,
                                                  mirror_y=mirror_y,
                                                  target_y_offset=target_y_offset)

            for index, row in df_annotation.iterrows():
                # Get the center coordinates of the annotation
                center_x = row["Center X"]
                center_y = row["Center Y"]

                # Transform the coordinates
                transformed_x, transformed_y = transformer.transform_coordinates((center_x, center_y))

                # Crop the patch
                local_mask = crop_patch_png(mask, (int(transformed_x), int(transformed_y)), (1100, 1100))

                # Downsample patch
                #local_mask = binning_image(local_mask, 2)

                # Isolate the central glomerulus
                isolated_glom_mask = isolate_central_mask(cv2.convertScaleAbs(local_mask))

                # Load original image
                glom_id = row["ID"]

                # Save masked image
                output_path = f"{output_dir}/patch_p{patient}_s{stain}_i{glom_id}.png"
                plt.imsave(output_path, isolated_glom_mask, cmap='gray')

                del isolated_glom_mask
                del local_mask
                gc.collect()

        print(f"Processed {mask_path}")
        del mask
        gc.collect()