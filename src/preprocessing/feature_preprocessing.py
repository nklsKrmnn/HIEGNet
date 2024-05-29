import pandas as pd
import numpy as np

from src.preprocessing.preprocessing_constants import SCALER_OPTIONS

def feature_preprocessing(df: pd.DataFrame, train_indices: list, scaler: str = None) -> pd.DataFrame:
    """
    feature_preprocessing: preprocess the features in the dataframe using the specified scaler. Scaler is fitted to
    the train set only.

    :param df: (pd.DataFrame) the dataframe containing the features to be preprocessed
    :param train_indices: (list) the indices of the training samples
    :param scaler: (str) the name of the scaler to be used
    :return: (pd.DataFrame) the dataframe with the features preprocessed
    """
    if scaler not in SCALER_OPTIONS:
        raise ValueError(f"Scaler {scaler} not supported. Supported scalers are {list(SCALER_OPTIONS.keys())}")

    if scaler is not None:
        scaler = SCALER_OPTIONS[scaler]()
        scaler = scaler.fit(df.iloc[train_indices])
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        df_scaled = df

    return df_scaled

