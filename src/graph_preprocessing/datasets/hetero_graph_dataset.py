from typing import Final
import cv2
import pandas as pd
import numpy as np
import torch
import os

from torch_geometric.data import HeteroData

from graph_preprocessing.preprocessing_constants import SCALER_OPTIONS
from src.graph_preprocessing.datasets.dataset_utils.dataset_utils import get_train_val_test_indices, create_mask
from src.graph_preprocessing.graph_preprocessing.hetero_graph_processing import drop_cell_glom_edges, create_cell_glom_edges
from src.graph_preprocessing.graph_preprocessing.knn_graph_constructor import graph_construction, graph_connection
from src.graph_preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
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
        # Fit scaler to edge attributes
        if patient in self.train_patients:
            self.create_and_fit_edge_scaler(data, ('glomeruli', 'to', 'glomeruli'), patient)


        y = self.create_targets(df_patient, data.target_labels)

        # Generate stratified train, val and test indices
        indices = get_train_val_test_indices(y,
                                             test_split=self.test_split,
                                             val_split=self.val_split,
                                             random_seed=self.random_seed,
                                             is_test_patient=(patient in self.test_patients),
                                             is_val_patient=(patient in self.validation_patients),
                                             glom_indices=df_patient['glom_index'].values,
                                             set_indices_path=self.set_indices_path,
                                             split_action=self.split_action)
        train_indices, val_indices, test_indices = indices

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
            current_cell_type = self.cell_types[i]

            # Add scaler for cell type if not existing
            if current_cell_type not in self.node_scalers.keys():
                self.node_scalers.update({current_cell_type: SCALER_OPTIONS[self.preprocessing_params['scaler']]()})

            # Gat all nodes for that patient
            df_cell_nodes = df_cell_nodes[df_cell_nodes['patient'] == patient].reset_index(drop=True)

            # Create connections between cells
            edge_type = (current_cell_type, 'to', current_cell_type)
            cell_coords = df_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
            cell_edge_index, cell_edge_weight = graph_construction(cell_coords, **self.cell_graph_params)
            data[edge_type].edge_index = torch.tensor(cell_edge_index, dtype=torch.long)
            data[edge_type].edge_attr = torch.tensor(cell_edge_weight, dtype=torch.float).unsqueeze(1)
            # Fit scaler to edge attributes
            self.create_and_fit_edge_scaler(data, edge_type, patient)

            # Create connections to other cell types
            for j, df_other_cell_nodes in enumerate(list_cell_nodes):
                edge_type = (current_cell_type, 'to', self.cell_types[j])
                if i == j:
                    # Skip same cell type
                    continue

                df_other_cell_nodes = df_other_cell_nodes[df_other_cell_nodes['patient'] == patient].reset_index(
                    drop=True)

                other_cell_coords = df_other_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
                other_cell_edge_index, other_cell_edge_weight = graph_connection(cell_coords, other_cell_coords,
                                                                                 **self.cell_graph_params)
                data[edge_type].edge_index = torch.tensor(other_cell_edge_index,dtype=torch.long)
                data[edge_type].edge_attr = torch.tensor(other_cell_edge_weight,dtype=torch.float).unsqueeze(1)

                # Fit scaler to edge attributes
                self.create_and_fit_edge_scaler(data, edge_type, patient)

            # Create connections between cells and glomeruli
            cell_glom_edge_index, cell_glom_edge_distances = create_cell_glom_edges(df_cell_nodes,
                                                                                    df_patient['glom_index'])
            # Add cell to glom edge index and weights to data
            edge_type = (current_cell_type, 'to', 'glomeruli')
            data[edge_type].edge_index = cell_glom_edge_index
            data[edge_type].edge_attr = cell_glom_edge_distances
            # Fit scaler to edge attributes
            self.create_and_fit_edge_scaler(data, edge_type, patient)

            # Add same edges back from glom to cells
            if self.cell_glom_connection:
                data['glomeruli', 'to', current_cell_type].edge_index = torch.stack(
                    [cell_glom_edge_index[1], cell_glom_edge_index[0]], dim=0).long()
                data['glomeruli', 'to', current_cell_type].edge_attr = cell_glom_edge_distances
                # Fit scaler to edge attributes
                self.create_and_fit_edge_scaler(data, ('glomeruli', 'to', current_cell_type), patient)


            # Get node features
            data[current_cell_type].x = self.create_features(df=df_cell_nodes,
                                                             train_indices=list(range(0, len(df_cell_nodes))),
                                                             feature_list=[c for c in df_cell_nodes.columns if
                                                                           c in self.cell_features],
                                                             preprocessing_params={'scaler': None})
            # Partial fit of scaler to node features
            if patient in self.train_patients:
                self.node_scalers[current_cell_type].partial_fit(data[current_cell_type].x)

        return data

    def create_and_fit_edge_scaler(self, data: torch.tensor, edge_type: tuple, patient: str) -> None:
        """
        Create and fit the edge scaler for the given edge type.

        Creates new scaler only if not existing yet. Fits it to the edge attributes of the given edge type if patient is not a test patient.

        :param data: Edge attributes.
        :param edge_type: Edge type to scale.
        :param patient: Patient number.
        :return: -
        """
        # Check if edges exist
        if len(data[edge_type].edge_attr) > 0:
            # Create new scaler if not existing yet
            if edge_type not in self.edge_scalers.keys():
                self.edge_scalers.update({edge_type: SCALER_OPTIONS[self.preprocessing_params['scaler']]()})
            # Fit scaler to edge attributes if patient is not in test set
            if patient in self.train_patients:
                self.edge_scalers[edge_type].partial_fit(data[edge_type].edge_attr)

    def scale_features(self, data_objects: dict) -> dict:
        """
        Scales glomeruli features across all graphs.

        For glomeruli, the node features and train masks are first concatenated and the sclaer is fitted to the train
        set only. Then, the features are scaled for all graphs. For all other cell types, the features are scaled
        with their respective scalers fitted partially before in the create_graph_object method.

        :param data_objects: Dict with graph data objects
        :return: Dict with graph data object, where x for glomeruli is scaled
        """
        # TODO: Implement partial scaling for glomeruli as well
        # Get all glomeruli features
        all_glom_features = torch.cat([data['glomeruli'].x for data in data_objects.values()], dim=0)
        all_train_masks = torch.cat([data.train_mask for data in data_objects.values()], dim=0)

        # Fit the scaler to the whole train set
        scaler = SCALER_OPTIONS[self.preprocessing_params['scaler']]()
        scaler.fit(all_glom_features[all_train_masks])

        # Scale the features for all graphs
        for data in data_objects.values():
            data['glomeruli'].x = torch.tensor(scaler.transform(data['glomeruli'].x), dtype=torch.float)

        # Scale node features for other cell types
        for key in data_objects:
            data = data_objects[key]
            for cell_type in self.cell_types:
                if (cell_type in data.node_types) and (cell_type != 'glomeruli'):
                    data_transformed = torch.tensor(self.node_scalers[cell_type].transform(data[cell_type].x),
                                                     dtype=torch.float)
                    data_objects[key][cell_type].x = data_transformed

        # Scale edge attributes for all edges
        for data in data_objects.values():
            for edge_type in self.edge_scalers.keys():
                try:
                    data[edge_type].edge_attr = torch.tensor(self.edge_scalers[edge_type].transform(data[edge_type].edge_attr),
                                                            dtype=torch.float)
                except:
                    pass

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
