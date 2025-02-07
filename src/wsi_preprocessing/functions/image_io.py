import cv2
import skimage
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path: str) -> np.array:
    """
    Load an image from a png file.
    :param image_path: Path to the image file (png)
    :return: Image as numpy array
    """
    image = skimage.io.imread(image_path)
    return image


def get_paths(input_folder: str, file_types: list[str] | str = ['.png', '.svs']) -> list[str]:
    """
    Get image paths in input folder.
    :param input_folder: Path to the input folder
    :return: List of image paths
    """
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(tuple(file_types)):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_csv_paths(input_folder: str) -> list[str]:
    """
    Get csv files in input folder.
    :param input_folder: Path to the input folder
    :return: List of image paths
    """
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.csv')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def read_svs_image(imagePath: str, lod: int) -> np.array:
    """
    read_svs_image: read a 2D RGB svs image at a given lod and return a numpy array

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """
    import openslide

    Image.MAX_IMAGE_PIXELS = None

    scaleFactor = 2 ** lod
    imageSvs = openslide.OpenSlide(imagePath)

    vecOfScales = np.asarray(imageSvs.level_downsamples)

    if np.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:  #  If the scale exist
        #  Find the index of the given scaleFactor in the vector of scales
        level = np.where(vecOfScales.astype(int) == scaleFactor)[0][0]
    else:
        string = [str(int(a)) for a in imageSvs.level_downsamples]
        raise ValueError(
            'The svs image does not contain an image with scaleFactor %i \n scales are: %s' % (scaleFactor, string))

    image = imageSvs.read_region((0, 0), level, imageSvs.level_dimensions[level])
    image = image.convert('RGB')
    imageSvs.close()

    return np.asarray(image)


def read_svs_patch(imagePath: str, lod: int, location: tuple[int, int], size: tuple[int, int]) -> np.array:
    """
    Read a patch from a svs image at a given lod and return a numpy array

    The scaling factor is 2^lod. The location is the center of the patch. The center location and size are according
    to the given lod.

    :param imagePath: (string) the path of the svs image
    :param lod: (int) the level of detail to be loaded
    :param location: (tuple) the center location of the patch to be read
    :param size: (tuple) the size of the patch to be read
    :return: (numpy.array) the image converted to a numpy array (size dependent upon the lod)
    """
    import openslide

    Image.MAX_IMAGE_PIXELS = None

    scaleFactor = 2 ** lod
    imageSvs = openslide.OpenSlide(imagePath)

    vecOfScales = np.asarray(imageSvs.level_downsamples)

    if np.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:  #  If the scale exist
        #  Find the index of the given scaleFactor in the vector of scales
        level = np.where(vecOfScales.astype(int) == scaleFactor)[0][0]
    else:
        string = [str(int(a)) for a in imageSvs.level_downsamples]
        raise ValueError(
            'The svs image does not contain an image with scaleFactor %i \n scales are: %s' % (scaleFactor, string))

    # Transform center coordinates to top left coordinates for the read_region functions
    # Also invert the y axis
    location = (location[0] - size[0] // 2, imageSvs.level_dimensions[level][1] - location[1] - size[1] // 2)

    # Apply scaling to size and location
    size = (size[0] // scaleFactor, size[1] // scaleFactor)
    location = (location[0] // scaleFactor, location[1] // scaleFactor)

    image = imageSvs.read_region(location, level, size)
    image = image.convert('RGB')
    imageSvs.close()

    return np.asarray(image)


def crop_patch_png(image: np.array, location: tuple[int, int], size: tuple[int, int]) -> np.array:
    """
    Crop a patch from an image at a given location and size.

    :param image: (numpy.array) the image to crop the patch from
    :param location: (tuple) the center location of the patch to be cropped
    :param size: (tuple) the size of the patch to be cropped
    :return: (numpy.array) the cropped patch
    """
    # Extract the patch
    x, y = location
    h, w = size
    patch = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]

    return patch


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


def upsampling_image(image: np.ndarray, upsampling_factor: int, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Upsampling the image by a factor.

    Args:
        image (np.ndarray): The input image.
        upsampling_factor (int): The factor to upsample the image by.
        interpolation (int): The interpolation method to use.

    Returns:
        np.ndarray: The upsampled image.
    """
    image = cv2.resize(image, (image.shape[1] * upsampling_factor, image.shape[0] * upsampling_factor),
                       interpolation=interpolation)
    return image


def isolate_central_mask(image: np.array) -> np.array:
    # Binarize the image
    binary_image = (image > 0).astype(np.uint8)

    # Isolate central glomerulus
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    center = (image.shape[0] // 2, image.shape[0] // 2)

    # Find the contour that covers the central pixel
    central_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            central_contour = contour
            break

    # If central contour is not on central pixel check for closest contour
    if central_contour is None:
        min_dist = -1000
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, center, True)
            if dist > min_dist:
                min_dist = dist
                central_contour = contour

    # Create a mask for the central cell structure
    mask = np.zeros_like(image)
    if central_contour is not None:
        cv2.drawContours(mask, [central_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to isolate the central cell structure
    isolated_image = cv2.bitwise_and(image, image, mask=mask)

    return isolated_image


def main():
    # test upsampling
    image = np.zeros((100, 100))
    image[10:20, 10:20] = 1

    image_upsampled = upsampling_image(image, 2, interpolation=cv2.INTER_NEAREST)

    print("stop")


if __name__ == "__main__":
    main()
