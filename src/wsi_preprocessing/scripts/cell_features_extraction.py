import argparse
from typing import Final
import numpy as np
import pandas as pd

from src.wsi_preprocessing.functions.image_io import get_paths
from src.wsi_preprocessing.functions.instance_segmentation_features import calc_cell_counts, calc_cell_areas, calc_cell_counts_by_radius
from src.wsi_preprocessing.functions.path_io import extract_patient_id, extract_index

def setup_parser() -> argparse.ArgumentParser:
    """
    Sets up the argument parser with the "cell_type" arguments.

    Returns:
        ArgumentParser: The argument parser with the added arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--celltype",
        "-t",
        type=str,
        required=True,
        help="Cell type to extract features for ('M0' or 'tcell').",
    )
    return parser

parser = setup_parser()
args, _ = parser.parse_known_args()

PROJECT: Final[str] = "EXC"
CELL_TYPE: Final[str] = args.celltype

input_dir = f"/home/dascim/data/3_extracted_features/{PROJECT}/masks_cellpose/{CELL_TYPE}/"
output_dir = f"/home/dascim/data/3_extracted_features/{PROJECT}/"

input_mask_paths = get_paths(input_dir, ".npy")

batch_size = 100
n_batches = len(input_mask_paths) // batch_size

cell_counts = []
cell_areas = []
cell_counts_by_radius = []

for i in range(n_batches + 1):
    print(f'Batch: {i}')
    masks = [np.load(mask_path) for mask_path in input_mask_paths[i*batch_size:(i+1)*batch_size]]

    cell_counts.extend(calc_cell_counts(masks))
    cell_areas.extend(calc_cell_areas(masks))
    radii = [300, 600, 900, 1200]
    cell_counts_by_radius.extend(calc_cell_counts_by_radius(masks, radii))

df_features = pd.DataFrame(input_mask_paths, columns=["mask_path"])
df_features["cell_counts"] = cell_counts
df_features["cell_ares"] = cell_areas
df_counts_by_radius = pd.DataFrame(cell_counts_by_radius)
df_counts_by_radius.columns = [f"cell_counts_radius_{r}" for r in radii]
df_features = pd.concat([df_features, df_counts_by_radius], axis=1)

df_features['patient'] = df_features['mask_path'].apply(lambda p: extract_patient_id(p))
df_features['glom_index'] = df_features['mask_path'].apply(lambda p: extract_index(p))
df_features.drop(columns=["mask_path"], inplace=True)

df_features.sort_values(by=['glom_index'], inplace=True)

#df_features.to_csv(f"{output_dir}/cell_features.csv", index=False)
