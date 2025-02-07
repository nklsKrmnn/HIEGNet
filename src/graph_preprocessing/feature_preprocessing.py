import os

import pandas as pd
import numpy as np
import torch

from src.graph_preprocessing.preprocessing_constants import SCALER_OPTIONS


def feature_preprocessing(df: pd.DataFrame,
                          feature_list: list,
                          train_indices: list,
                          scaler: str = None) -> torch.Tensor:
    """
    feature_preprocessing: preprocess the features in the dataframe using the specified scaler. Scaler is fitted to
    the train set only.

    :param df: (pd.DataFrame) the dataframe containing the features to be preprocessed
    :param feature_list: (list) the list of features to be used
    :param train_indices: (list) the indices of the training samples
    :param scaler: (str) the name of the scaler to be used
    :return: (pd.DataFrame) the dataframe with the features preprocessed
    """
    df = df[feature_list]

    if (scaler not in SCALER_OPTIONS) and (scaler is not None):
        raise ValueError(f"Scaler {scaler} not supported. Supported scalers are {list(SCALER_OPTIONS.keys())}")

    if scaler is not None:
        scaler = SCALER_OPTIONS[scaler]()
        scaler = scaler.fit(df.iloc[train_indices])
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        df_scaled = df

    # Convert true false to 1 and 0
    df_scaled = df_scaled.replace({True: 1, False: 0})

    x = torch.tensor(df_scaled.to_numpy(), dtype=torch.float)

    return x


def get_image_paths(df: pd.DataFrame,
                    feature_list: list[str],
                    root_dir: str) -> list:
    """
    get_image_paths: get the paths of all images in the input directory with the specified extension

    :param input_dir: (str) the directory containing the images
    :param extension: (str) the extension of the images
    :return: (list) the list of paths to the images
    """
    return [[root_dir + df[path].iloc[i] for path in feature_list] for i in
            range(df.shape[0])]
