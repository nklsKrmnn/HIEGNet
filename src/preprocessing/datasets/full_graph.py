from typing import Final, Any
import cv2
import pandas as pd
import torch
import os
from src.preprocessing.datasets.hetero_hybrid_graph_dataset import HeteroHybridGraphDataset
from src.preprocessing.feature_preprocessing import feature_preprocessing
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")

class FullGraphDataset(HeteroHybridGraphDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_file_paths(self):
        return [self.feature_file_path] + super().input_file_paths #TODO implement dict

    def create_features(self,
                        df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str]) -> list[list[str | Any]]:

        # Check for each feature if feature column in dataframe is numeric
        numeric_feature = [ft for ft in feature_list if df[ft].apply(pd.to_numeric, errors='coerce').notnull().all().all()]

        # Preprocess features
        x_numeric = feature_preprocessing(df, numeric_feature, train_indices, **self.preprocessing_params)

        # Get paths to images
        not_numeric_feature = [ft for ft in feature_list if ft not in numeric_feature]
        if len(not_numeric_feature) > 0:
            x_image = self.create_image_input_tensor(df, not_numeric_feature)
            return (x_numeric, x_image)
        else:
            return x_numeric

    @property
    def image_size(self):
        first_graph = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))

        # Load images, if image features are used and transform to tensor
        img = cv2.imread(first_graph['glomeruli'].x[1][0][0])
        return img.shape[0]

    def get(self, idx):
        # Load graph
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images, if image features are used and transform to tensor
        item['glomeruli_image'].x = self.get_images_from_paths(idx, item['glomeruli'].x[1])
        item['glomeruli'].x = item['glomeruli'].x[0]

        return item
