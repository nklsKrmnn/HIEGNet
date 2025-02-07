import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.wsi_preprocessing.functions.coordinate_transformer import CoordinateTransformater
from src.wsi_preprocessing.functions.image_io import crop_patch_png, binning_image, get_paths
import os
from src.wsi_preprocessing.functions.path_io import get_path_up_to, extract_patient_id

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")

STAINS = [25]
PATIENTS = ['002', '003', '004']

for stain in STAINS:
    # Get dir for masks
    dir_path = f'{ROOT_DIR}/data/1_cytomine_downloads/EXC/roi/{stain}/masks/glomeruli/'

    # Get all files in the directory
    file_paths = get_paths(dir_path, ".png")
    output_dir = f"{ROOT_DIR}/data/2_images_preprocessed/EXC/patches_glom_isolated/{stain}"

    for file in file_paths:
        # Load mask
        mask = plt.imread(file)
        patient = file.split("/")[-1].split("_")[2]

        if patient in PATIENTS:
            # Import annotations
            df_annotation = pd.read_csv(f'{ROOT_DIR}/data/1_cytomine_downloads/EXC/annotations/{stain}/annotations_{patient}_{stain}.csv')
            df_annotation = df_annotation[df_annotation["Term"] != "Tissue"]

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
                patch = crop_patch_png(mask, (int(transformed_x), int(transformed_y)), (1100,1100))

                # Downsample patch
                patch = binning_image(patch, 2)

                # Load original image
                glom_id = row["ID"]
                image_path = f"{ROOT_DIR}/data/2_images_preprocessed/EXC/patches_low_resolution/{stain}/patch_p{patient}_s{stain}_i{glom_id}.png"
                image_patch = plt.imread(image_path)

                # mask glomerulus
                masked_image = image_patch * patch[:,:,np.newaxis]

                # Show the masked image
                #plt.imshow(masked_image)
                #plt.show()

                # Save masked image
                output_path = f"{output_dir}/patch_p{patient}_s{stain}_i{glom_id}.png"
                plt.imsave(output_path, masked_image)



        print(f"Processed {file}")
        mask = None
