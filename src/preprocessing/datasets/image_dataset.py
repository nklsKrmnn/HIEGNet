import os
from typing import Final, Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import compute_class_weight
from torchvision.io import read_image
import numpy as np

from src.preprocessing.datasets.dataset_utils.dataset_utils import list_annotation_file_names, \
    get_train_val_test_indices
from src.preprocessing.datasets.dataset_utils.image_utils import load_images
from src.preprocessing.datasets.hybrid_graph_dataset import HybridGraphDataset
from src.preprocessing.feature_preprocessing import get_image_paths
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class GlomImageDataset(HybridGraphDataset):
    def __init__(self,
                 annotations_path,
                 image_file_path: str,
                 feature_list: list,
                 random_seed: int = 42,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 test_patients: list[str] = [],
                 hot_load: bool = False,
                 onehot_targets: bool = True):

        self.test_split = test_split
        self.val_split = validation_split
        self.train_patients = train_patients
        self.test_patients = test_patients
        self.path_file = ROOT_DIR + image_file_path
        self.annotations_paths = list_annotation_file_names(ROOT_DIR + annotations_path)
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.hot_load = hot_load
        self.feature_list = feature_list
        if self.hot_load:
            self.x_hot = []
        self.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

        # Attributes to be set when data is processed
        self.targets = None
        self.img_paths = None
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
            df = df[df['patient'].isin(self.train_patients)].reset_index(drop=True)

        # Make set split per patient to be consistent with the graph datasets when selecting multiple patients and to
        # enable to test for generalisation of a different patient
        train_indices, val_indices, test_indices = [], [], []
        for patient in df['patient'].unique():
            indices = df[df['patient'] == patient].index
            idx = get_train_val_test_indices(y=self.targets[indices],
                                             test_split=self.test_split,
                                             val_split=self.val_split,
                                             random_seed=self.random_seed,
                                             is_val_patient=False,
                                             is_test_patient=(patient in self.test_patients))
            train_indices += [indices[i] for i in idx[0]]
            val_indices += [indices[i] for i in idx[1]]
            test_indices += [indices[i] for i in idx[2]]


        self.img_paths = self.create_features(df, [], self.feature_list)
        self.indices = (train_indices, val_indices, test_indices)

        # Safe patient ids for visualisation purposes
        self.patients = df['patient']


    def create_features(self,
                        df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str],
                        preprocessing_params: dict = None) -> list[list[str | Any]]:

        # Get paths to images
        x = get_image_paths(df, feature_list, ROOT_DIR)

        # Save images in instance variable if hot load is enabled
        if self.hot_load:
            self.x_hot = load_images(x)

        return x

    @property
    def image_size(self):
        image = read_image(self.img_paths[0][0])
        return image.shape[1]

    def get_set_indices(self) -> tuple[np.array, np.array, np.array]:
        return self.indices

    def get_class_weights(self) -> torch.Tensor:
        """
        Get the class weights for the dataset.

        The class weights are
        calculated based on the distribution of the target labels in the dataset.
        """
        # Labels of train data
        train_indices = self.get_set_indices()[0]
        y_labels = self.targets[train_indices]

        if self.onehot_targets:
            y_labels = y_labels.argmax(dim=1).numpy()
            label_classes = np.unique(y_labels)
        else:
            y_labels = y_labels.numpy()
            label_classes = np.unique(y_labels)

        class_weights = compute_class_weight('balanced',
                                             classes=label_classes,
                                             y=y_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        return class_weights_tensor

    def create_folds(self, n_folds: int) -> None:
        """
        Creates n fold for cross validation.

        Applies a stratified k fold split on dataset excluding the test set and saves the indices of the folds into a dict.

        :param n_folds: Number of cross validation folds
        :return: None
        """
        # Get train and val indices
        train_indices, val_indices, _ = self.get_set_indices()

        indices = np.array(train_indices + val_indices)

        # Get targets
        y_labels = self.targets[indices]
        if self.onehot_targets:
            y_labels = y_labels.argmax(dim=1).numpy()
        else:
            y_labels = y_labels.numpy()

        # Create folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

        for fold, (train_indices, val_indices) in enumerate(skf.split(indices, y_labels)):
            self.folds[fold] = (list(indices[train_indices]), list(indices[val_indices]))

    def activate_fold(self, fold: int) -> None:
        """
        Activates a specific fold for cross validation.

        :param fold: The fold to activate
        :return: None
        """
        self.indices = (self.folds[fold][0], self.folds[fold][1], self.indices[2])

    def __len__(self):
        return len(self.img_paths)

    def transform_image(self, image):
        return image / 255.0

    def __getitem__(self, idx):
        if self.hot_load:
            image = self.x_hot[idx]
        else:
            image = load_images([self.img_paths[idx]])[0]
        label = self.targets[idx]
        return image, label
