import cv2
import numpy as np
import os

def binning_image(image: np.ndarray, binning_factor: int) -> np.ndarray:
    """
    Binning the image by a factor.

    Args:
        image (np.ndarray): The input image.
        binning_factor (int): The factor to bin the image by.

    Returns:
        np.ndarray: The binned image.
    """
    image = cv2.resize(image, (image.shape[1] // binning_factor, image.shape[0] // binning_factor))
    return image


path_original_images = "/home/dascim/data/2_images_preprocessed/EXC/patches/25"
path_binned_images = "/home/dascim/data/2_images_preprocessed/EXC/patches_low_resolution/25"

# Create output directory if it does not exist
os.makedirs(path_binned_images, exist_ok=True)

# Loop through all images in the original images directory
for image_name in os.listdir(path_original_images):
    # Read the image
    image = cv2.imread(os.path.join(path_original_images, image_name))

    # Binning the image by a factor of 2
    binned_image = binning_image(image, 4)

    # Save the binned image
    cv2.imwrite(os.path.join(path_binned_images, image_name), binned_image)