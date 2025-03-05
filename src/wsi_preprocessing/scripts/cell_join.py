import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

# ignore warnings
import warnings

from src.wsi_preprocessing.functions.path_io import get_path_up_to

warnings.filterwarnings('ignore')

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

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")
CELL_TYPE = args.celltype
PATCH_SIZE = 2200

path = f"{ROOT_DIR}/data/3_extracted_features/EXC/cell_nodes/{CELL_TYPE}_cell_nodes_prejoin.csv"
df = pd.read_csv(f"{ROOT_DIR}/data/3_extracted_features/EXC/cell_nodes/{CELL_TYPE}_cell_nodes_prejoin_56.csv")

df['center_distance'] = np.sqrt((df['center_x_local'] - PATCH_SIZE//2 )**2 + (df['center_y_local'] - PATCH_SIZE//2 )**2)

temp_df = pd.DataFrame({
    'glom_index': df['associated_glom'],
    'distance': df['center_distance'],
    'is_in_glom': df['is_in_glom']
})

df['associated_glomeruli'] = temp_df.apply(lambda row: [{'glom_index': row['glom_index'],
                                                         'distance': row['distance'],
                                                         'is_in_glom': row['is_in_glom']}],
                                           axis=1)

df["center_x_glom"] = df["center_x_global"] - df["center_x_local"]
df["center_y_glom"] = df["center_y_global"] + df["center_y_local"]

df_glomeruli = df[['associated_glom', 'center_x_glom', 'center_y_glom', 'patient']].groupby("associated_glom").mean()

for i, glom in tqdm(df_glomeruli.iterrows(), total=len(df_glomeruli)):
    # Find all cells associated with the current glomerulus
    df_cells = df[df["associated_glom"] == i]

    # Find all glomeruli that are close to the current one
    glom_x = glom["center_x_glom"]
    glom_y = glom["center_y_glom"]
    df_glomeruli["dist"] = np.sqrt(
        (df_glomeruli["center_x_glom"] - glom_x) ** 2 + (df_glomeruli["center_y_glom"] - glom_y) ** 2)
    close_glomeruli = df_glomeruli[df_glomeruli["dist"] <= PATCH_SIZE]

    for j, close_glom in close_glomeruli.iterrows():
        # Skip the current glomerulus and glomeruli from other patients
        if (i == j) or close_glom['patient'] != glom['patient']:
            continue

        # Find all cells associated with the close glomeruli that are close to the current one
        df_close_cells = df[df["associated_glom"] == j]

        for q, cell in df_cells.iterrows():
            # Skip if the cell is already removed
            if q not in df.index:
                continue

            # Find all cells that are close to the current one
            cell_x = cell["center_x_global"]
            cell_y = cell["center_y_global"]
            df_close_cells["dist"] = np.sqrt(
                (df_close_cells["center_x_global"] - cell_x) ** 2 + (df_close_cells["center_y_global"] - cell_y) ** 2)
            potential_same_cells = df_close_cells[(df_close_cells["dist"] < 10) & (df_close_cells.index != q)]

            if len(potential_same_cells) > 0:
                # Make sure that the closest cell is the first in the list
                potential_same_cells = potential_same_cells.sort_values("dist")

                if len(potential_same_cells) > 1:
                    print(f"Multiple close cells found for cell {q} in glomerulus {i} and glomerulus {j}!")

                # Add the close cells to the associated_glomeruli list
                df.loc[q, "associated_glomeruli"].append(
                    {'glom_index': j,
                     'distance': potential_same_cells.iloc[0]["center_distance"],
                     'is_in_glom': potential_same_cells.iloc[0]["is_in_glom"]})

                # Remove the close cells from the dataframe
                if potential_same_cells.index.values[0] in df.index.values:
                    df.drop(potential_same_cells.index[0], inplace=True)

df.to_csv(f"{ROOT_DIR}/data/3_extracted_features/EXC/cell_nodes/{CELL_TYPE}_cell_nodes_56.csv", index=False)
# pickle results
df.to_pickle(f"{ROOT_DIR}/data/3_extracted_features/EXC/cell_nodes/{CELL_TYPE}_cell_nodes_56.pkl")
