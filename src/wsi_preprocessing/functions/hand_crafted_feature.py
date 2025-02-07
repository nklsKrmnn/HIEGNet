import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_shape_based_features(binary_image: np.array) -> dict:
    """
    Extract shape-based features from a binary image.

    Parameters
    ----------
    binary_image : np.array
        Binary image.

    Returns
    -------
    dict
        Features
    """

    # Extract contour
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # Extract features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    ellipse = cv2.fitEllipse(contour)
    aspect_ratio = ellipse[1][1] / ellipse[1][0]
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1]) ** 2)

    features = {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'eccentricity': eccentricity
    }

    return features


def extract_texture_based_features(image: np.array) -> dict:
    """
    Extract texture-based features from an image.

    Parameters
    ----------
    image : np.array
        Image.

    Returns
    -------
    dict
        Features
    """

    # Extract GLCM
    glcm = graycomatrix(image, [1], [0, np.pi / 2], 256, symmetric=True, normed=True)

    # Extract features
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    features = {
        'contrast': contrast,
        'correlation': correlation,
        'energy': energy,
        'homogeneity': homogeneity
    }

    # Extract lbp histogram
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    #lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 7), density=True)

    # Get count per uniform pattern
    pattern, count = np.unique(lbp, return_counts=True)

    # Remove pattern 8 as this is the background
    pattern = np.delete(pattern, [-2], None)
    count = np.delete(count, [-2], None)

    # Normalize count
    count = count / count.sum()

    features.update({f'lbp_uniform_{int(pat)}': value for pat, value in zip(pattern, count)})

    return features
