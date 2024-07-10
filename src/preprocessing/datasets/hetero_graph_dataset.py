from typing import Final
import cv2
import pandas as pd
import numpy as np
import pickle
import torch
import os

from sklearn.utils import compute_class_weight
from torch_geometric.data import Data, HeteroData
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.preprocessing.datasets.dataset_utils.dataset_utils import get_train_val_test_indices
from src.preprocessing.feature_preprocessing import feature_preprocessing
from src.preprocessing.graph_preprocessing.knn_graph_constructor import knn_graph_construction, \
    knn_weighted_graph_construction, graph_construction
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class HeteroGraphDataset(GlomGraphDataset):
    def __init__(self, **kwargs):
        self.cell_types = kwargs.pop('cell_types')
        self.cell_graph_params = kwargs.pop('cell_graph')

        self.cell_node_files = [os.path.join(kwargs.pop('cell_node_dir_path'), f"{cell_type}_cell_nodes.pkl") for
                                cell_type in
                                self.cell_types]

        super().__init__(**kwargs)

    def create_graph_object(self, df_patient) -> HeteroData:
        """
        Create the graph object from the raw data.

        The raw data is loaded from the raw file and a graph is constructed from the point cloud with the knn graph
        algorithm. Afterward, node features and note-wise targets for node classification are added. A train and test
        mask is added to the graph data object.

        :return: The graph data object
        """
        patient = df_patient['patient'].iloc[0]

        data = HeteroData()

        # Target labels
        data.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

        # Create the graph from point cloud and generate the edge index
        coords = df_patient[['Center X', 'Center Y']].to_numpy()
        edge_index, edge_weights = graph_construction(coords, **self.glom_graph)
        data[('glomeruli', 'to', 'glomeruli')].edge_index = torch.tensor(edge_index, dtype=torch.long)
        data[('glomeruli', 'to', 'glomeruli')].edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        y = self.create_targets(df_patient, data.target_labels)

        # Generate stratified train, val and test indices
        train_indices, val_indices, test_indices = get_train_val_test_indices(y, self.test_split,
                                                                              self.val_split,
                                                                              self.random_seed,
                                                                              self.test_patients,
                                                                              self.validation_patients)

        data.train_mask = self.create_mask(len(y), train_indices)
        data.val_mask = self.create_mask(len(y), val_indices)
        data.test_mask = data.val_mask if (self.test_split == 0.0) and (self.test_patients == []) else self.create_mask(
            len(y), test_indices)

        # Create the node features in tensor
        data["glomeruli"].x = self.create_feature_tensor(df_patient, train_indices, self.feature_list)

        data.y = y

        data.glom_indices = torch.tensor(df_patient['glom_index'].values)
        data.coords = torch.tensor(coords, dtype=torch.float)

        list_cell_nodes = [pd.read_pickle(file) for file in self.cell_node_files]

        # Add other cells to graph
        for i, df_cell_nodes in enumerate(list_cell_nodes):
            df_cell_nodes = df_cell_nodes[df_cell_nodes['patient'] == patient].reset_index(drop=True)

            # Create connections between cells
            cell_coords = df_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
            cell_edge_index, cell_edge_weight = graph_construction(cell_coords, **self.cell_graph_params)
            data[self.cell_types[i], 'to', self.cell_types[i]].edge_index = torch.tensor(cell_edge_index,
                                                                                         dtype=torch.long)
            data[self.cell_types[i], 'to', self.cell_types[i]].edge_attr = torch.tensor(cell_edge_weight,
                                                                                        dtype=torch.float).unsqueeze(1)

            # Create connections between cells and glomeruli
            df_cell_nodes['cell_row'] = np.arange(df_cell_nodes.shape[0])
            df_cells_exploded = df_cell_nodes.explode('associated_glomeruli')
            df_cells_exploded['glom_index'] = df_cells_exploded['associated_glomeruli'].apply(lambda d: d['glom_index'])
            df_cells_exploded['distance'] = df_cells_exploded['associated_glomeruli'].apply(lambda d: d['distance'])
            df_glom_connection = df_patient['glom_index'].to_frame()
            df_glom_connection['glom_row'] = np.arange(df_glom_connection.shape[0])
            df_glom_connection = df_cells_exploded.merge(df_glom_connection, right_on='glom_index',
                                                         left_on='glom_index', how='inner')
            cell_glom_edge_index = (df_glom_connection['cell_row'], df_glom_connection['glom_row'])
            cell_glom_edge_weights = torch.tensor(df_glom_connection['distance'].values, dtype=torch.float).unsqueeze(1)

            data[self.cell_types[i], 'to', 'glomeruli'].edge_index = torch.tensor(cell_glom_edge_index,
                                                                                  dtype=torch.long)
            data[self.cell_types[i], 'to', 'glomeruli'].edge_attr = cell_glom_edge_weights

            # Get node features
            data[self.cell_types[i]].x = self.create_feature_tensor(df=df_cell_nodes,
                                                                    train_indices=list(range(0, len(df_cell_nodes))),
                                                                    feature_list=[c for c in df_cell_nodes.columns if
                                                                                  c.endswith('_node_feature')])

        return data

    def get(self, idx):
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images, if image features are used and transform to tensor
        if not isinstance(item['glomeruli'].x[0], torch.Tensor):
            images = []
            for paths in item.x:
                slices = []
                for slice in paths:
                    slices.append(cv2.imread(slice))
                img = np.concatenate(slices, axis=2)
                images.append(img)
            item.x = np.array(images)
            item.x = torch.tensor(item.x, dtype=torch.float)
            item.x = item.x.permute(0, 3, 1, 2)

        return item
