from typing import Final, Any
import cv2
import pandas as pd
import numpy as np
import torch
import os
import time

from src.preprocessing.datasets.dataset_utils.image_utils import load_images
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.preprocessing.datasets.hetero_graph_dataset import HeteroGraphDataset
from src.preprocessing.datasets.hybrid_graph_dataset import HybridGraphDataset
from src.preprocessing.feature_preprocessing import get_image_paths, feature_preprocessing
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")

class HeteroHybridGraphDataset(HeteroGraphDataset, HybridGraphDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_features(self,
                        df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str]) -> list[list[str | Any]]:

        # Check if feature columns in dataframe are numeric
        if df[feature_list].apply(pd.to_numeric, errors='coerce').notnull().all().all():
            # Preprocess features
            x = feature_preprocessing(df, feature_list, train_indices, **self.preprocessing_params)
        else:
            # Get paths to images
            x = self.create_image_input_tensor(df, feature_list)
        return x

    @property
    def image_size(self):
        first_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))

        # Load images, if image features are used and transform to tensor
        img = cv2.imread(first_graph['glomeruli'].x[0][0])
        return img.shape[0]

    def get(self, idx):
        # Load graph
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        for node_type in item.node_types:
            # Check if node feature are numeric
            if not isinstance(item[node_type].x, torch.Tensor):
                # Load images
                item[node_type].x = self.get_images_from_paths(idx, item[node_type].x)

        return item
