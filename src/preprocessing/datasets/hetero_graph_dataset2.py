from typing import Final
import cv2
import pandas as pd
import numpy as np
import torch
import os
from itertools import product

from torch_geometric.data import HeteroData

from preprocessing.datasets.hetero_graph_dataset import HeteroGraphDataset
from preprocessing.preprocessing_constants import SCALER_OPTIONS
from src.preprocessing.datasets.dataset_utils.dataset_utils import get_train_val_test_indices, create_mask
from src.preprocessing.graph_preprocessing.hetero_graph_processing import drop_cell_glom_edges, create_cell_glom_edges
from src.preprocessing.graph_preprocessing.knn_graph_constructor import graph_construction, graph_connection
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


class HeteroUnifiedGraphDataset(HeteroGraphDataset):

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
            df_cell_nodes["cell_type"] = current_cell_type

            # Create connections to other cell types
            for j, df_other_cell_nodes in enumerate(list_cell_nodes):
                if i == j:
                    # Skip same cell type
                    continue

                other_node_type = self.cell_types[j]

                df_other_cell_nodes = df_other_cell_nodes[df_other_cell_nodes['patient'] == patient].reset_index(
                    drop=True)
                df_other_cell_nodes["cell_type"] = other_node_type
                df_cell_nodes = pd.concat([df_cell_nodes, df_other_cell_nodes], ignore_index=True)

            # Create connections between cells
            edge_type = (current_cell_type, 'to', current_cell_type)
            cell_coords = df_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
            cell_edge_index, cell_edge_weight = graph_construction(cell_coords, **self.cell_graph_params)

            # Create edge type as combination of cell types
            node_type_combinations = list(product(self.cell_types, repeat=2))
            edge_types = [(nt1, 'to', nt2) for nt1, nt2 in node_type_combinations]

            for edge_type in edge_types:
                node_type1, _, node_type2 = edge_type
                mask_source_nodes = df_cell_nodes["cell_type"].iloc[cell_edge_index[0]] == node_type1
                mask_target_nodes = df_cell_nodes["cell_type"].iloc[cell_edge_index[1]] == node_type2
                edge_index_mask = mask_source_nodes.values & mask_target_nodes.values

                edge_index = (cell_edge_index[0][edge_index_mask], cell_edge_index[1][edge_index_mask])
                edge_weight = cell_edge_weight[edge_index_mask]

                data[edge_type].edge_index = torch.tensor(edge_index, dtype=torch.long)
                data[edge_type].edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)

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

