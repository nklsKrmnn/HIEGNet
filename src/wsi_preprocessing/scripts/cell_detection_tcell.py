import os
from typing import Final
from src.wsi_preprocessing.functions.instance_segmentation_features import cell_segmentation
from src.wsi_preprocessing.functions.path_io import get_path_up_to

STAINING: Final[str] = "25"
PROJECT: Final[str] = "EXC"
CELL_TYPE: Final[str] = "tcell"
MODEL_NAME: Final[str] = "241213_tcell_model_between"
# Channels. Determined in the creation of the florescent images
CHAN: Final[int] = 3  # Blue for cytoplasm
CHAN2: Final[int] = 0  # No channel for nuclei
# Default values from cellpose
FLOW_THRESHOLD: Final[float] = 0.4
CELLPROB_THRESHOLD: Final[float] = 0.0

ROOT_DIR = get_path_up_to(os.path.abspath(__file__), "repos")



def main() -> None:
    """
    Main functions for cell detection

    Iterates over all images in the input directory and applies cellpose to each image. Segmented masks are saved in the
    output directory.
    :return: None
    """

    input_dir = "/home/niklas/2_images_preprocessed/25/"
    input_dir = f"{ROOT_DIR}/data/2_images_preprocessed/{PROJECT}/patches_florescent/{STAINING}/"
    output_dir = "/home/niklas/3_extracted_features"
    output_dir = f"{ROOT_DIR}/data/3_extracted_features"

    print(input_dir)

    cell_segmentation(input_dir=input_dir,
                      output_dir=output_dir,
                      project=PROJECT,
                      cell_type=CELL_TYPE,
                      model_name=MODEL_NAME)


if __name__ == "__main__":
    main()
