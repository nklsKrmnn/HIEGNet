import os
from typing import Final, Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import compute_class_weight
from torchvision.io import read_image
import numpy as np

from preprocessing.datasets.image_dataset import GlomImageDataset
from src.preprocessing.datasets.dataset_utils.dataset_utils import list_annotation_file_names, \
    get_train_val_test_indices
from src.preprocessing.datasets.dataset_utils.image_utils import load_images
from src.preprocessing.datasets.hybrid_graph_dataset import HybridGraphDataset
from src.preprocessing.feature_preprocessing import get_image_paths, feature_preprocessing
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class TabularDataset(GlomImageDataset):
    def __init__(self,
                 annotations_path,
                 feature_file_path: str,
                 feature_list: list,
                 random_seed: int = 42,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 test_patients: list[str] = [],
                 split_action:str = 'load',
                 preprocessing_params: dict = None,
                 set_indices_path: str = "/repos/histograph/data/input/set_indices/test15_val15",
                 onehot_targets: bool = True):

        self.test_split = test_split
        self.val_split = validation_split
        self.train_patients = train_patients
        self.test_patients = test_patients
        self.split_action = split_action
        self.set_indices_path = ROOT_DIR + set_indices_path
        self.path_file = ROOT_DIR + feature_file_path
        self.annotations_paths = list_annotation_file_names(ROOT_DIR + annotations_path)
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.feature_list = feature_list
        self.preprocessing_params = preprocessing_params
        self.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

        # Attributes to be set when data is processed
        self.targets = None
        self.indices = None
        self.patients = None
        self.folds = {}

    def process(self) -> None:

        print('[Dataset]: Processing data')

        df = pd.read_csv(self.path_file)
        df_annotations = pd.concat([pd.read_csv(path) for path in self.annotations_paths])
        df = pd.merge(df, df_annotations, left_on="glom_index", right_on="ID", how="left")

        # Sort df the same order as the graph dataset
        sorted_indices = pd.read_csv(ROOT_DIR + "/data/3_extracted_features/EXC/glom_index_order.csv")["glom_index"]
        df = df.set_index("glom_index").loc[sorted_indices].reset_index()

        # Drop rows where feature or image path is missing (most likely because no match through slices)
        df.dropna(subset=self.feature_list, inplace=True)

        # Create target labels
        self.targets = self.create_targets(df, self.target_labels)

        # Select only the patients that are in the train_patients list
        if len(self.train_patients) > 0:
            df = df[df['patient'].isin(self.train_patients + self.test_patients)].reset_index(drop=True)

        # Make set split per patient to be consistent with the graph datasets when selecting multiple patients and to
        # enable to test for generalisation of a different patient
        train_indices, val_indices, test_indices = [], [], []
        for patient in df['patient'].unique():
            indices = df[df['patient'] == patient].index
            idx = get_train_val_test_indices(y=self.targets[indices],
                                             test_split=self.test_split,
                                             val_split=self.val_split,
                                             random_seed=self.random_seed,
                                             is_val_patient=(patient in self.train_patients),
                                             is_test_patient=(patient in self.test_patients),
                                             glom_indices=df[df['patient'] == patient]['glom_index'].values,
                                             set_indices_path=self.set_indices_path,
                                             split_action=self.split_action)
            train_indices += [indices[i] for i in idx[0]]
            val_indices += [indices[i] for i in idx[1]]
            test_indices += [indices[i] for i in idx[2]]


        self.input = self.create_features(df, train_indices, self.feature_list, self.preprocessing_params)
        self.indices = (train_indices, val_indices, test_indices)

        # Safe patient ids for visualisation purposes
        self.patients = df['patient']


    def create_features(self,
                        df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str],
                        preprocessing_params: dict = None) -> list[list[str | Any]]:
        # TODO use from parent
        x = feature_preprocessing(df, feature_list, train_indices, **preprocessing_params)

        return x

    @property
    def image_size(self):
        return None


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.targets[idx]
        return input, label
