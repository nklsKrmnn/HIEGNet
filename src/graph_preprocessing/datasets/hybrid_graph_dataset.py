from typing import Final, Any
import cv2
import pandas as pd
import numpy as np
import torch
import os
import time

from src.graph_preprocessing.datasets.dataset_utils.image_utils import load_images
from src.graph_preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.graph_preprocessing.feature_preprocessing import get_image_paths
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class HybridGraphDataset(GlomGraphDataset):

    def __init__(self, image_file_path: str, hot_load: bool = False, **kwargs):
        self.image_file_path = ROOT_DIR + image_file_path
        self.hot_load = hot_load
        if self.hot_load:
            self.x_hot = []

        super().__init__(**kwargs)

    @property
    def input_file_paths(self) -> list[str]:
        return [self.image_file_path]

    def create_image_input_tensor(self, df: pd.DataFrame, feature_list: list[str]) -> list:
        # Get paths to images
        x = get_image_paths(df, feature_list, ROOT_DIR)

        # Save images in instance variable if hot load is enabled
        if self.hot_load:
            images = load_images(x)
            self.x_hot.append(images)

        return x

    def create_features(self,
                        df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str],
                        preprocessing_params: dict) -> list[list[str | Any]]:
        return self.create_image_input_tensor(df, feature_list)

    @property
    def image_size(self):
        first_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))

        # Load images, if image features are used and transform to tensor
        img = cv2.imread(first_graph.x[0][0])
        return img.shape[0]

    def get_images_from_paths(self, idx, x):
        # Load images
        if self.hot_load:
            images = self.x_hot[idx]
        else:
            images = load_images(x)
        return images

    def get(self, idx):
        # Load graph
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images
        item.x = self.get_images_from_paths(idx, item.x)

        return item
