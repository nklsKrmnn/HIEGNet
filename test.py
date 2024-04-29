import numpy as np
import math

OFFSET_THRESHOLD = 10

def transform_coordinates(pix_point: tuple[float, float]) -> tuple[float, float]:
    """
    Converts pixel coordinates to robot coordinates.

    Args:
        pix_point (tuple[float, float]): pixel coordinates

    Returns:
        tuple[float, float]: Robot coordinate (x-value, y-value)
    """
    x_pix, y_pix = pix_point

    # The rotation matrix rotates counterclockwise at positive angles.
    DEGREE = 90
    # The sine and cosine function expect radians
    # https://docs.python.org/3/library/math.html
    RAD = math.radians(DEGREE)

    # 2.358 pixels correspond to 1 mm
    SCALE = (1/16)

    # Coordinates relative to robot origin (0,0)
    TARGET_X_OFFSET = 4709
    TARGET_Y_OFFSET = 0

    # Homogeneous coordinates
    input_vector = np.array([x_pix, y_pix, 1])

    # Translation by the distance between camera origin and robot origin.
    camera_pos_trans_matrix = np.array([[1, 0, TARGET_X_OFFSET],
                                        [0, 1, TARGET_Y_OFFSET],
                                        [0, 0, 1]])
    '''
    # Y-axis is mirrored over X-axis.
    mirror_y_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])
                                '''

    # The passed points are rotated clockwise by 90 degrees around the origin.
    rotation_matrix = np.array([[math.cos(RAD), -math.sin(RAD), 0],
                                [math.sin(RAD), math.cos(RAD), 0],
                                [0, 0, 1]])

    # The pixel values are converted to millimeters.
    scale_matrix = np.array([[SCALE, 0, 0],
                             [0, SCALE, 0],
                             [0, 0, 1]])

    # The passed points are rotated clockwise by 90 degrees around the origin.
    rotated_input = np.matmul(rotation_matrix, input_vector)
    # Y-axis is mirrored over X-axis.
    #mirrored_input = np.matmul(mirror_y_matrix, rotated_input)
    # The pixel values are converted to millimeters.
    scaled_input = np.matmul(scale_matrix, rotated_input)
    # The distance between camera origin and robot origin is added.
    positioned_input = np.matmul(camera_pos_trans_matrix, scaled_input)
    # Only the first two components of the vector are used.
    result = (positioned_input[0], positioned_input[1])

    # The calculated robot coordinate is returned.
    return result


class CoordinateTransformation:
    def __init__(self, magnification: float,
                 rotation: int,
                 crop_bottom: bool = False,
                 crop_left: bool = False,
                 mirror_x: bool = False,
                 mirror_y: bool = False,
                 target_x_offset: int = 0,
                 target_y_offset: int = 0):
        self.scale = magnification
        self.rotation = rotation
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.target_x_offset = target_x_offset
        self.target_y_offset = target_y_offset

    def calculate_offset(self, origin_point: tuple[float, float], target_point: tuple[int, int]):

        # Get transformation results without offset
        transformation_result = self.transform_coordinates(origin_point)

        # Calculate offset
        x_offset = target_point[0] - int(transformation_result[0])
        y_offset = target_point[1] - int(transformation_result[1])

        # Set offset to 0 if it is above the threshold
        self.target_x_offset = x_offset if abs(x_offset) > OFFSET_THRESHOLD else 0
        self.target_y_offset = y_offset if abs(y_offset) > OFFSET_THRESHOLD else 0

        print(f'[COORD_TRANSFORM] - Calculated offset: ({self.target_x_offset}, {self.target_y_offset})')

    def transform_coordinates(self, point):
        x_pix, y_pix = point

        # Homogeneous coordinates
        input_vector = np.array([x_pix, y_pix, 1])

        # Translation by the distance between camera origin and robot origin.
        camera_pos_trans_matrix = np.array([[1, 0, self.target_x_offset],
                                            [0, 1, self.target_y_offset],
                                            [0, 0, 1]])

        # Y-axis is mirrored over X-axis.
        mirror_x = -1 if self.mirror_x else 1
        mirror_y = -1 if self.mirror_y else 1
        mirror_y_matrix = np.array([[mirror_y, 0, 0],
                                    [0, mirror_x, 0],
                                    [0, 0, 1]])

        # Transform rotation to radians
        rotation_rad = math.radians(self.rotation)

        # The passed points are rotated clockwise by give degrees around the origin.
        rotation_matrix = np.array([[math.cos(rotation_rad), -math.sin(rotation_rad), 0],
                                    [math.sin(rotation_rad), math.cos(rotation_rad), 0],
                                    [0, 0, 1]])

        # The pixel values are converted to millimeters.
        scale_matrix = np.array([[self.scale, 0, 0],
                                 [0, self.scale, 0],
                                 [0, 0, 1]])

        # The passed points are rotated clockwise by 90 degrees around the origin.
        rotated_input = np.matmul(rotation_matrix, input_vector)
        # Y-axis is mirrored over X-axis.
        mirrored_input = np.matmul(mirror_y_matrix, rotated_input)
        # The pixel values are converted to millimeters.
        scaled_input = np.matmul(scale_matrix, rotated_input)
        # The distance between camera origin and robot origin is added.
        positioned_input = np.matmul(camera_pos_trans_matrix, scaled_input)
        # Only the first two components of the vector are used.
        result = (positioned_input[0], positioned_input[1])

        # The calculated robot coordinate is returned.
        return result

# Example usage
if __name__ == "__main__":

    origin_points = [[16038.24, 69265.17],
                     [20133.03, 47612.88]]  # Origin points
    target_points = [[380, 1001],
                     [1733, 1257]]  # Target points




    # The rotation matrix rotates counterclockwise at positive angles.
    DEGREE = 90

    # 2.358 pixels correspond to 1 mm
    SCALE = (1 / 16)

    # Coordinates relative to robot origin (0,0)
    TARGET_X_OFFSET = 0
    TARGET_Y_OFFSET = 0

    ct = CoordinateTransformation(magnification=SCALE,
                                  rotation=DEGREE,
                                  target_x_offset=TARGET_X_OFFSET,
                                  target_y_offset=TARGET_Y_OFFSET)

    # Test transformation of a point
    calibration_point = (5723.02, 56453.76)
    calibration_target = (1181, 357)

    ct.calculate_offset(calibration_point, calibration_target)

    point_to_transform = (16038.24, 69265.17)
    #[5723.02, 56453.76
    #transformed_point = ct.transform_point(point_to_transform)
    transformed_point = ct.transform_coordinates(point_to_transform)
    print("Transformed Point:", transformed_point)
