from typing import Final
import cv2
import pandas as pd
import numpy as np
import torch
import os

from torch_geometric.data import HeteroData

from preprocessing.preprocessing_constants import SCALER_OPTIONS
from src.preprocessing.datasets.dataset_utils.dataset_utils import get_train_val_test_indices, create_mask
from src.preprocessing.graph_preprocessing.hetero_graph_processing import drop_cell_glom_edges, create_cell_glom_edges
from src.preprocessing.graph_preprocessing.knn_graph_constructor import graph_construction, graph_connection
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class HeteroGraphDataset(GlomGraphDataset):
    def __init__(self,
                 send_msg_from_glom_to_cell: bool = False,
                 **kwargs):
        self.cell_types = kwargs.pop('cell_types')
        self.cell_graph_params = kwargs.pop('cell_graph')
        self.cell_features = kwargs.pop('cell_features')
        self.cell_glom_connection = send_msg_from_glom_to_cell

        cell_node_dir_path = ROOT_DIR + kwargs.pop('cell_node_dir_path')

        self.cell_node_files = [os.path.join(cell_node_dir_path, f"{cell_type}_cell_nodes.pkl") for
                                cell_type in
                                self.cell_types]

        super().__init__(**kwargs)

    def create_graph_object(self, df_patient, patient) -> HeteroData:
        """
        Create the graph object from the raw data.

        The raw data is loaded from the raw file and a graph is constructed from the point cloud with the knn graph
        algorithm. Afterward, node features and note-wise targets for node classification are added. A train and test
        mask is added to the graph data object.

        :return: The graph data object
        """

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
        train_indices, val_indices, test_indices = get_train_val_test_indices(y,
                                                                              test_split=self.test_split,
                                                                              val_split=self.val_split,
                                                                              random_seed=self.random_seed,
                                                                              is_test_patient=(
                                                                                      patient in self.test_patients),
                                                                              is_val_patient=(
                                                                                      patient in self.validation_patients))

        data.train_mask = create_mask(len(y), train_indices)
        data.val_mask = create_mask(len(y), val_indices)
        data.test_mask = data.val_mask if (self.test_split == 0.0) and (self.test_patients == []) else create_mask(
            len(y), test_indices)

        # Create the node features in tensor
        # No scaling here to scale later with all graphs
        data["glomeruli"].x = self.create_features(df_patient, train_indices, self.feature_list,
                                                   preprocessing_params={'scaler': None})

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

            # Create connections to other cell types
            for j, df_other_cell_nodes in enumerate(list_cell_nodes):
                if i == j:
                    # Skip same cell type
                    continue

                df_other_cell_nodes = df_other_cell_nodes[df_other_cell_nodes['patient'] == patient].reset_index(drop=True)

                other_cell_coords = df_other_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
                other_cell_edge_index, other_cell_edge_weight = graph_connection(cell_coords, other_cell_coords,
                                                                                 **self.cell_graph_params)
                data[self.cell_types[i], 'to', self.cell_types[j]].edge_index = torch.tensor(other_cell_edge_index,
                                                                                             dtype=torch.long)
                data[self.cell_types[i], 'to', self.cell_types[j]].edge_attr = torch.tensor(other_cell_edge_weight,
                                                                                            dtype=torch.float).unsqueeze(
                    1)

            # Create connections between cells and glomeruli
            cell_glom_edge_index, cell_glom_edge_distances = create_cell_glom_edges(df_cell_nodes, df_patient['glom_index'])

            # Add cell to glom edge index and weights to data
            data[self.cell_types[i], 'to', 'glomeruli'].edge_index = cell_glom_edge_index
            data[self.cell_types[i], 'to', 'glomeruli'].edge_attr = cell_glom_edge_distances

            # Add same edges back from glom to cells
            if self.cell_glom_connection:
                data['glomeruli', 'to', self.cell_types[i]].edge_index = torch.stack(
                    [cell_glom_edge_index[1], cell_glom_edge_index[0]], dim=0).long()
                data['glomeruli', 'to', self.cell_types[i]].edge_attr = cell_glom_edge_distances

            # Get node features
            data[self.cell_types[i]].x = self.create_features(df=df_cell_nodes,
                                                              train_indices=list(range(0, len(df_cell_nodes))),
                                                              feature_list=[c for c in df_cell_nodes.columns if
                                                                            c in self.cell_features],
                                                              preprocessing_params=self.preprocessing_params)

        return data

    def scale_glomeruli(self, data_objects: dict) -> dict:
        """
        Scales glomeruli features across all graphs.

        Unpacks all feature tensors from the data objects, fits the scaler to the whole train set and scales the features
        for all graphs.

        :param data_objects: Dict with graph data objects
        :return: Dict with graph data object, where x for glomeruli is scaled
        """

        # Get all glomeruli features
        all_glom_features = torch.cat([data['glomeruli'].x for data in data_objects.values()], dim=0)
        all_train_masks = torch.cat([data.train_mask for data in data_objects.values()], dim=0)

        # Fit the scaler to the whole train set
        scaler = SCALER_OPTIONS[self.preprocessing_params['scaler']]()
        scaler.fit(all_glom_features[all_train_masks])

        # Scale the features for all graphs
        for data in data_objects.values():
            data['glomeruli'].x = torch.tensor(scaler.transform(data['glomeruli'].x), dtype=torch.float)

        return data_objects

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
