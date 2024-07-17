from typing import Final, Any
import cv2
import pandas as pd
import numpy as np
import torch
import os
import time

from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")

class HybridGraphDataset(GlomGraphDataset):

    def __init__(self, hot_load: bool = False, **kwargs):
        self.hot_load = hot_load
        if self.hot_load:
            self.x_hot = []

        super().__init__(**kwargs)

    def create_feature_tensor(self,
                              df: pd.DataFrame,
                              train_indices: list[int],
                              feature_list: list[str]) -> list[list[str | Any]]:

        # Get paths to images
        x = [[ROOT_DIR + df[path].iloc[i] for path in self.feature_list] for i in
                 range(df.shape[0])]

        # Save images in instance variable if hot load is enabled
        if self.hot_load:
            images = self.load_images(x)
            self.x_hot.append(images)

        return x

    @property
    def image_size(self):
        first_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))

        # Load images, if image features are used and transform to tensor
        img = cv2.imread(first_graph.x[0][0])
        return img.shape[0]

    def load_images(self, paths) -> list[torch.tensor]:
        """
        Load images from paths.
        :param paths: List of tuples of paths to images.
        :return: List of images.
        """
        images = []
        for glom in paths:
            slices = []
            for slice in glom:
                slices.append(cv2.imread(slice))
            img = np.concatenate(slices, axis=2)
            images.append(img)

        # Transform to tensor
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float)
        images = images.permute(0, 3, 1, 2)

        return images

    def get(self, idx):
        # Load graph
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images
        if self.hot_load:
            images = self.x_hot[idx]
        else:
            images = self.load_images(item.x)
        item.x = images

        return item
