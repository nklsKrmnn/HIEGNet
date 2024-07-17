import os
from typing import Final, Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torchvision.io import read_image
import numpy as np

from src.preprocessing.datasets.dataset_utils.dataset_utils import list_annotation_file_names
from src.preprocessing.datasets.hybrid_graph_dataset import HybridGraphDataset
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class GlomImageDataset(HybridGraphDataset):
    def __init__(self,
                 annotations_path,
                 image_paths_file,
                 feature_list: list,
                 random_seed: int = 42,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 hot_load: bool = False,
                 onehot_targets:bool=True):

        self.test_split = test_split
        self.val_split = validation_split
        self.train_patients = train_patients
        self.path_file = image_paths_file
        self.annotations_paths = list_annotation_file_names(annotations_path)
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.hot_load = hot_load
        self.feature_list = feature_list
        if self.hot_load:
            self.x_hot = []


    def process(self) -> None:

        print('[Dataset]: Processing data')

        df = pd.read_csv(self.path_file)
        df_annotations = pd.concat([pd.read_csv(path) for path in self.annotations_paths])
        df = pd.merge(df, df_annotations, left_on="glom_index", right_on="ID", how="left")

        # Drop rows where feature or image path is missing (most likely because no match through slices)
        df.dropna(subset=self.feature_list, inplace=True)

        self.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']
        self.targets = self.create_targets(df, self.target_labels)
        self.img_paths = self.create_feature_tensor(df, [], self.feature_list)

    def create_feature_tensor(self,
                              df: pd.DataFrame,
                              train_indices: list[int],
                              feature_list: list[str]) -> list[list[str | Any]]:

        # Get paths to images
        x = [[ROOT_DIR + df[path].iloc[i] for path in self.feature_list] for i in
                 range(df.shape[0])]

        # Save images in instance variable if hot load is enabled
        if self.hot_load:
            self.x_hot=self.load_images(x)

        return x

    @property
    def image_size(self):
        image = read_image(self.img_paths[0][0])
        return image.shape[1]

    def get_set_indices(self):
        # get indices of train and test patients
        dataset_size = len(self)

        # random split
        indices = list(range(dataset_size))
        if self.test_split > 0:
            train_indices, test_indices = train_test_split(indices, test_size=self.test_split, random_state=self.random_seed,
                                                           stratify=self.targets)
        else:
            train_indices = indices
            test_indices = []
        val_split_correction = self.test_split * self.val_split
        train_indices, validation_indices = train_test_split(train_indices, test_size=self.val_split - val_split_correction,
                                                             random_state=self.random_seed,
                                                             stratify=self.targets[train_indices])
        if test_indices == []:
            test_indices = validation_indices

        return train_indices, validation_indices, test_indices

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

    def __len__(self):
        return len(self.img_paths)

    def transform_image(self, image):
        return image / 255.0

    def __getitem__(self, idx):
        if self.hot_load:
            image = self.x_hot[idx]
        else:
            image = self.load_images([self.img_paths[idx]])[0]
        label = self.targets[idx]
        return image, label