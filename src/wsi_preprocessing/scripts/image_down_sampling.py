import cv2
import os
from typing import Final
from tqdm import tqdm

from functions.image_io import binning_image

PATH_ORIGINAL_IMAGES: Final[str] = "/home/dascim/data/2_images_preprocessed/EXC/patches/25"
PATH_BINNED_IMAGES: Final[str] = "/home/dascim/data/2_images_preprocessed/EXC/patches_testing/25"
BINNING_FACTOR: Final[int] = 16

# Create output directory if it does not exist
os.makedirs(PATH_BINNED_IMAGES, exist_ok=True)

# Loop through all images in the original images directory
for image_name in tqdm(os.listdir(PATH_ORIGINAL_IMAGES)):
    # Read the image
    image = cv2.imread(os.path.join(PATH_ORIGINAL_IMAGES, image_name))

    # Binning the image by a factor of 4
    binned_image = binning_image(image, BINNING_FACTOR)

    # Save the binned image
    cv2.imwrite(os.path.join(PATH_BINNED_IMAGES, image_name), binned_image)