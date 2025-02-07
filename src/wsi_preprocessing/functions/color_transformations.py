import numpy as np
from skimage.color import rbd_from_rgb, separate_stains
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt


DEBUG_FLOR_IM = False
DEBUG_MACRO_IM = False
DEBUG_TCELL_IM = False


def transform_to_florescent(image: np.array) -> np.array:
    """
    Transform an image in np format in staining 68 (code:25) to a florescent image. Macrophages are in blue and
    T-cells in green.
    :param image: Image to transform
    :return: Florescent image
    """
    # Separate stains
    stain_channels = separate_stains(image, rbd_from_rgb)

    # Create florescent image
    red = np.zeros_like(stain_channels[:, :, 0])
    green = rescale_intensity(stain_channels[:, :, 0], out_range=(0, 1),
                              in_range=(0, np.percentile(stain_channels[:, :, 0], 99)))
    blue = rescale_intensity(stain_channels[:, :, 2], out_range=(0, 1),
                             in_range=(0, np.percentile(stain_channels[:, :, 2], 99)))

    # Cast the two channels into an RGB image, as the blue and green channels
    # Convert to ubyte for easy saving as image to local drive
    flor_image = img_as_ubyte(np.dstack((red, green, blue)))  # DAB in green and H in Blue

    if DEBUG_FLOR_IM:
        plt.figure()
        plt.imshow(flor_image)
        plt.axis("off")
        plt.show()

    return flor_image

def create_cell_masks(image: np.array) -> np.array:
    """
    Create masks for macrophages and T-cells.
    :param image: Image to create masks
    :return: Masks for macrophages and T-cells
    """
    # Separate stains
    stain_channels = separate_stains(image, rbd_from_rgb)

    macro_channel = stain_channels[:, :, 2]
    tcells_channel = stain_channels[:, :, 0]

    if DEBUG_MACRO_IM:
        plt.figure()
        plt.imshow(macro_channel)
        plt.figure()
        plt.imshow(image)
        plt.show()

    if DEBUG_TCELL_IM:
        plt.figure()
        plt.imshow(tcells_channel)
        plt.figure()
        plt.imshow(image)
        plt.show()

    # Create masks
    thresh_m = 0.05
    thresh_t = 0.1

    macro_mask = (macro_channel > thresh_m).astype(np.uint8)
    tcell_mask = (tcells_channel > thresh_t).astype(np.uint8)
    tcell_mask = tcell_mask - np.logical_and(tcell_mask, macro_mask)

    return macro_mask, tcell_mask