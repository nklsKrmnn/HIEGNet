import os

import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import torch
from tqdm import tqdm

from src.wsi_preprocessing.functions.image_io import get_paths, load_image


def threshold_segementation(image: np.array,
                            thres_intensity: int = 200,
                            thres_area: int = 80,
                            channel: int = 1) -> np.array:
    """
    Segments cells in an image using a fixed threshold. The green channel is isolated and a fixed threshold is applied to
    the image. Contours are found and masks are created for contours above a threshold size.

    :param image: Image
    :param thres_intensity: Threshold intensity for target channel
    :param thres_area: Threshold area for cells
    :param channel: Target channel to segment in image. Defaults to 1 (green channel)
    :return: Mask
    """

    # Isolate target channel
    image = image[:, :, channel-1]

    # Apply threshold
    image_fixed_threshold = cv2.threshold(image, thres_intensity, 255, cv2.THRESH_BINARY)[1]

    # Binarize image
    image_bin = image_fixed_threshold > 0

    # Find contours
    contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove small contours
    contours = [contour for contour in contours if cv2.contourArea(contour) > thres_area]

    # Create masks for contours above threshold size
    mask = np.zeros_like(image_bin, dtype=np.uint8)
    for i, contour in enumerate(contours):
        cv2.drawContours(mask, [contour], -1, i + 1, -1)

    return mask


def cell_segmentation(input_dir: str,
                      output_dir: str,
                      project: str,
                      cell_type: str,
                      model_name: str,
                      chan: int = 3,
                      chan2: int = 0,
                      flow_threshold: float = 0.4,
                      cellprob_threshold: float = 0.0,
                      thres_intensity: int = 200,
                      thres_area: int = 80) -> list:
    """
    Segments cells in images using cellpose and saves the masks in the output directory.

    Iterates over all images in the input directory. Applies the cellpose model to each image and saves the segmented
    masks in the output directory.

    :param input_dir: Input directory containing the images
    :param output_dir: Output directory to save the masks for each image
    :param project: EXC or EXA (for path)
    :param cell_type: Type of cells that are segmented (relevant for file naming)
    :param model_name: Name of the cellpose model
    :param chan: Channel for segmentation model
    :param chan2: Channel two for segmentation model
    :param flow_threshold:
    :param cellprob_threshold:
    :return:
    """

    # Get all images in the input folder
    input_image_paths = get_paths(input_dir)
    print(f"Found {len(input_image_paths)} images in {input_dir}")

    # Load model
    if model_name is not None:
        model_path = output_dir + f"/cell_segmentation_models/{model_name}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = models.CellposeModel(gpu=True,
                                     pretrained_model=model_path,
                                     device=device)

        # Use model diameter
        diameter = model.diam_labels

    image_paths = [image_path for image_path in input_image_paths]

    masks = []

    # save masks as png
    for image_path, path in tqdm(zip(image_paths, input_image_paths), total=len(image_paths)):

        image = load_image(image_path)

        if model_name is not None:
            mask, flow, style = model.eval(image,
                                           channels=[chan, chan2],
                                           diameter=diameter,
                                           flow_threshold=flow_threshold,
                                           cellprob_threshold=cellprob_threshold
                                           )
        else:
            mask = threshold_segementation(image, thres_intensity, thres_area, channel=chan)

        mask_name = os.path.basename(path)

        mask_name = f"{cell_type}_mask_" + "_".join(mask_name.split("_")[-3:])
        mask_path = f"{output_dir}/{project}/masks_cellpose/{cell_type}/{mask_name}"
        plt.imsave(mask_path, mask, cmap='gray')
        np.save(mask_path.replace(".png", ".npy"), mask)
        #masks.append(mask)

    return masks


def calc_cell_counts(masks: list[np.array]) -> list[int]:
    return [np.unique(mask).size for mask in masks]


def calc_cell_areas(masks: list[np.array]) -> list[float]:
    return [np.sum((mask > 0)) for mask in masks]


def calc_cell_counts_by_radius(masks: list[np.array], radii: list[int]) -> list:
    count_by_radius = []
    for i, mask in enumerate(masks):
        mask_center = np.array(mask.shape) / 2

        binary_mask = (mask > 0).astype(int)
        cells = np.unique(mask)
        centers = center_of_mass(binary_mask, mask, range(1, len(cells) + 1))

        cell_center_distances = [np.linalg.norm(center - mask_center) for center in centers]

        count_by_radius.append({})
        for radius in radii:
            count_by_radius[i][radius] = len([dist for dist in cell_center_distances if dist < radius])

    return count_by_radius
