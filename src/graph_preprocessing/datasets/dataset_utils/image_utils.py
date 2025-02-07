import cv2
import numpy as np
import torch


def load_images(paths) -> list[torch.tensor]:
    """
    Load images from paths.
    :param paths: List of tuples of paths to images.
    :return: List of images.
    """
    images = []
    for glom in paths:
        slices = []
        for slice in glom:
            slices.append(cv2.imread(slice))
        img = np.concatenate(slices, axis=2)
        images.append(img)

    # Transform to tensor
    images = np.array(images)
    images = torch.tensor(images, dtype=torch.float)
    images = images.permute(0, 3, 1, 2)

    return images
