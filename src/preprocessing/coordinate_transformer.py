import numpy as np
import math
from typing import Final

OFFSET_THRESHOLD: Final[float] = 10.0

class CoordinateTransformater:
    def __init__(self, magnification: float,
                 rotation: int,
                 mirror_x: bool = False,
                 mirror_y: bool = False,
                 target_x_offset: int = 0,
                 target_y_offset: int = 0):
        self.scale = magnification
        self.rotation = rotation
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
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

    def match_coordinates(self, origin: np.array, target: np.array) -> list:

        # Transform the origin coordinates
        transformed_origin = np.array([self.transform_coordinates((x, y)) for x, y in origin])

        # Match to target coordinates with nearest neighbour
        matched_points = []
        for point in transformed_origin:
            distances = np.linalg.norm(target - point, axis=1)
            best_match = np.argmin(distances)
            best_match_distance = np.min(distances)

            if best_match_distance < 3:
                # Check if the target point is already matched
                i = 0
                stop_counter = 0
                while i < len(matched_points) and stop_counter < 5:
                    if matched_points[i][0] == best_match:
                        # If the target point is already matched, check if the new match is better
                        if matched_points[i][1] > best_match_distance:
                            # If the new match is better, replace the old match
                            matched_points[i] = (-1, np.inf)
                            i = len(matched_points)
                        else:
                            # If the old match is better, find the next best match
                            distances[best_match] = np.inf
                            best_match = np.argmin(distances)
                            best_match_distance = np.min(distances)
                            stop_counter += 1
                            i = 0
                    else:
                        i += 1

                # If the target point is matched, add to the list
                matched_points.append((np.argmin(distances), np.min(distances)))
            else:
                # If the target point is not matched, add to the list
                matched_points.append((-1, np.inf))

        # Return only the index matches
        return matched_points[:][0]

