import pandas as pd
import numpy as np
import pickle
import os
from os.path import isfile, join
from typing import Final

from src.preprocessing.coordinate_transformer import CoordinateTransformater
from src.utils.constants import REFERENCE_POINTS

RAW_DIR_PATH: Final[str] = 'data/input/raw'

# TODO: Implement patient and staining in config
#PATIENT: Final[list[str]] = ['001']
STAINING: Final[str] = '25'


def clean_indices(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Resets indices of a DataFrame and gives a warning, if any indices where missing
    :param df: DataFrame
    :param name: Name of the data, that is checked
    :return: DataFrame with cleaned indices
    """
    prior_max_index = df.index.max()
    df.reset_index(drop=True, inplace=True)
    posterior_max_index = df.index.max()

    if prior_max_index != posterior_max_index:
        print(
            f"[RAW_DATA_GENERATION] - Warning: {prior_max_index - posterior_max_index} indices were missing and have been reset in {name}.")

    return df


def generate_raw_data(input_paths: dict,
                      raw_file_name: str,
                      patient_list: list[str],
                      coordinate_transformation: dict = None):
    """
    Generate raw data file from input paths and feature list.

    Get DataFrames for matching results, extracted features, centroids and annotations. Merge them and apply
    coordinate transformation for the matching with the annotation. Afterward, the raw data is saved as a csv file.

    :param input_paths: dictionary of input paths
    :param raw_file_name: file name to save the raw data
    :param patient_list: list of features to be extracted
    :param coordinate_transformation: dictionary of coordinate transformation parameters
    :return: None
    """
    df_list = []
    for patient in patient_list:
        matching_file_name = f'matching_across_stainings_{patient}_from_segmentation.npy'
        centroids_file_name = f'IFTA_EXC_{patient}_{STAINING}_centroids_from_segmentation.npy' # TODO: check if semgentation is corret


        # Read matchings
        df_matchings = pd.read_pickle(input_paths['df_matchings'] + matching_file_name)
        df_matchings = clean_indices(df_matchings, name='Matchings')

        # Read all feature DataFrames from pickle files from directory feature_dir
        feature_dir = input_paths['feature_dir']
        feature_files = [f for f in os.listdir(feature_dir) if isfile(join(feature_dir, f))]
        feature_df_list = [pd.read_pickle(join(feature_dir, ft_file)) for ft_file in feature_files if patient in ft_file]

        # Concatenate all feature DataFrames
        df_features = pd.concat(feature_df_list, axis=1)
        df_features = clean_indices(df_features, name='Features')

        # Read centroids
        df_centroids = pd.read_pickle(input_paths['df_centroids'] + centroids_file_name)
        df_centroids = clean_indices(df_centroids, name="Centroids")

        # Get target df
        df_targets = pd.read_csv(input_paths['df_annotations'], delimiter=';')
        df_targets = df_targets[['Center X', 'Center Y', 'Term', 'Image filename']]

        # Get staining and patient from image filename
        df_targets["patient"] = df_targets['Image filename'].str.split('_').str[2]
        df_targets['staining'] = df_targets['Image filename'].str.split('_').str[-1]
        df_targets['staining'] = df_targets['staining'].str.split('.').str[0]
        df_targets.drop(columns=['Image filename'], inplace=True)

        # Filter by patient and staining
        df_targets = df_targets[(df_targets['patient'] == patient) & (df_targets['staining'] == STAINING)]

        # Merge centroids with matching results
        df = pd.merge(df_matchings, df_centroids, left_on="s1", right_index=True, how="left")

        n_pre_drop = df.shape[0]
        df.dropna(inplace=True)
        n_post_drop = df.shape[0]
        if n_post_drop < n_pre_drop:
            print(
                f"[RAW_DATA_GENERATION] - Warning: {n_pre_drop - n_post_drop} matchings are not used due to missing "
                f"centroids.")
        if len(df_centroids) > n_pre_drop:
            print(
                f"[RAW_DATA_GENERATION] - Warning: {len(df_centroids) - n_pre_drop} centroids are not used due to missing "
                f"matchings.")

        # Merge features into df
        df = pd.merge(df, df_features, left_index=True, right_index=True, how="left")

        n_pre_drop = df.shape[0]
        df.dropna(inplace=True)
        n_post_drop = df.shape[0]
        if n_post_drop < n_pre_drop:
            print(
                f"[RAW_DATA_GENERATION] - Warning: {n_pre_drop - n_post_drop} matchings are not used due to missing "
                f"feature entries.")
        if len(df_features) > n_pre_drop:
            print(
                f"[RAW_DATA_GENERATION] - Warning: {len(df_features) - n_pre_drop} feature samples are not used due to "
                f"missing matchings.")

        # Extract centroid coordinates
        df['centroid_x'] = df["centroid"].apply(lambda a: a[0])
        df['centroid_y'] = df["centroid"].apply(lambda a: a[1])

        # Apply coordinate transformation
        coord_tras_params = coordinate_transformation.copy()
        ref_point_origin = REFERENCE_POINTS[patient]['origin'] #tuple(coord_tras_params.pop('reference_point_annotation').values())
        ref_point_target = REFERENCE_POINTS[patient]['target'] #tuple(coord_tras_params.pop('reference_point_image').values())
        coord_transformer = CoordinateTransformater(**coord_tras_params)
        coord_transformer.calculate_offset(ref_point_origin, ref_point_target)

        # Get indices to match to df
        transformation_results = coord_transformer.match_coordinates(df_targets[['Center X', 'Center Y']].to_numpy(),
                                                                     df[['centroid_x', 'centroid_y']].to_numpy())
        # Add match index to df_targets
        df_targets['match_index'] = np.array(transformation_results)[:, 0]

        # Merge targets into df wit determined indeces
        df = pd.merge(df, df_targets, left_index=True, right_on='match_index', how='left')

        n_pre_drop = df.shape[0]
        df.dropna(inplace=True)
        n_post_drop = df.shape[0]
        if n_post_drop < n_pre_drop:
            print(
                f"[RAW_DATA_GENERATION] - Warning: {n_pre_drop - n_post_drop} matchings are not used due to missing "
                f"targets.")
        # Drop unnessecary columns from df_target
        df.drop(columns=['Center X', 'Center Y', 'match_index', 'staining'], inplace=True)

        # Append to list
        df_list.append(df)

    # Save as csv
    # TODO: fix conflic with root param in config
    df = pd.concat(df_list)
    path = os.path.join(os.getcwd(), RAW_DIR_PATH, raw_file_name)
    df.to_csv(path, index=False)
