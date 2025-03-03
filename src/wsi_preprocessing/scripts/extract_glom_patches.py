from src.wsi_preprocessing.functions.image_io import get_paths, read_svs_patch
import os
from typing import Final
import pandas as pd
import matplotlib.pyplot as plt

from wsi_preprocessing.functions.color_transformations import transform_to_florescent

CLASSES = ['Healthy','Dead','Sclerotic']
STAINING: Final[str] = "25"
PROJECT: Final[str] = "EXC"
LOD: Final[int] = 0
CROPPING_SIZE: Final[int] = 2200 #hight and width of the patch around a glom in pixels
patients = ['005','006']

input_dir = f"/home/dascim/data/1_cytomine_downloads/{PROJECT}"
output_dir = f"/home/dascim/data/2_images_preprocessed/{PROJECT}"

svs_image_dir = f"{input_dir}/svs_images/{STAINING}"
annotations_dir = f"{input_dir}/annotations/{STAINING}"

# Get all images in the input folder
input_image_paths = get_paths(svs_image_dir)

for image_path in input_image_paths:
    image_name = os.path.basename(image_path)
    patient = image_name.split("_")[2]

    if patient not in patients:
        continue

    annotation_file_name = f"annotations_{patient}_{STAINING}.csv"
    csv_path = f"{annotations_dir}/{annotation_file_name}"

    print(f"Image path: {image_path}")
    print(f"CSV path: {csv_path}")

    annotations = pd.read_csv(csv_path)
    annotations = annotations[annotations['Term'].apply(lambda t: t in CLASSES)].reset_index(drop=True)

    for index, row in annotations.iterrows():
        patch = read_svs_patch(image_path,
                               lod=LOD,
                               location=(int(row["Center X"]), int(row["Center Y"])),
                               size=(CROPPING_SIZE, CROPPING_SIZE)
                               )

        # Save patch in brightfield
        patch_name = f"patch_p{patient}_s{STAINING}_i{row['ID']}.png"
        patch_path = f"{output_dir}/patches/{STAINING}/{patch_name}"
        plt.imsave(patch_path, patch)

        # Save patch in florescent
        patch_florescent = transform_to_florescent(patch)
        patch_name = f"patch_flor_p{patient}_s{STAINING}_i{row['ID']}.png"
        patch_path = f"{output_dir}/patches_florescent/{STAINING}/{patch_name}"
        plt.imsave(patch_path, patch_florescent)

        print(f"Patch saved at {patch_path}")


    print("")
