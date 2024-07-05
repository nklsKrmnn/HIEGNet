import cv2
import pandas as pd
import numpy as np
import pickle
import torch
import os

from sklearn.utils import compute_class_weight
from torch_geometric.data import Data, HeteroData
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.preprocessing.feature_preprocessing import feature_preprocessing
from src.preprocessing.graph_preprocessing.knn_graph_constructor import knn_graph_construction, \
    knn_weighted_graph_construction
from src.utils.file_name_utils import get_glom_index
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset


class HeteroGraphDataset(GlomGraphDataset):
    def __init__(self, **kwargs):
        cell_node_dir_path = kwargs.pop('cell_node_dir_path')
        cell_types = kwargs.pop('cell_types')

        super().__init__(**kwargs)
        self.cell_node_dir_path = cell_node_dir_path
        self.cell_types = cell_types

    def process(self) -> None:
        """
        Process the raw data to the graph data format.

        The raw data is loaded from the raw file and a graph is constructed from the point cloud with the knn graph
        algorithm. Afterward, node features and note-wise targets for node classification are added. A train and test
        mask is added to the graph data object. Finally, the data object is saved to the processed directory.

        :return: None
        """
        df = pd.read_csv(self.raw_paths[0])
        df_annotations = pd.concat([pd.read_csv(path) for path in self.raw_paths[1:]])
        df = pd.merge(df, df_annotations, left_on="glom_index", right_on="ID", how="left")
        patients_in_raw_data = df['patient'].unique()

        cell_node_files = [os.path.join(self.cell_node_dir_path, f"{cell_type}_cell_nodes.csv") for cell_type in
                           self.cell_types]
        list_cell_nodes = [pd.read_csv(path) for path in cell_node_files]

        file_names = []

        for patient in patients_in_raw_data:
            df_patient = df[df['patient'] == patient]

            # Drop rows where feature or image path is missing (most likely because no match through slices)
            df_patient.dropna(subset=self.feature_list, inplace=True)

            # threshold for minimum number of data points and check if patient is in train or test set
            if (df_patient.shape[0] > 10) and (
                    patient in self.train_patients + self.validation_patients + self.test_patients):

                # Create the data object for each graph
                data = HeteroData()

                # Target labels
                data.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

                # Create the graph from point cloud and generate the edge index
                coords = df_patient[['Center X', 'Center Y']].to_numpy()
                adjectency_matrix = knn_graph_construction(coords, self.n_neighbours)
                edge_index = torch.tensor(np.argwhere(adjectency_matrix == 1).T, dtype=torch.long)
                data[('glomeruli', 'to', 'glomeruli')].edge_index = edge_index

                # Create the node features in tensor
                if self.path_image_inputs is None:
                    # Get numerical features
                    x = df_patient[self.feature_list]
                    x = torch.tensor(x.to_numpy(), dtype=torch.float)
                else:
                    # Get image paths
                    x = [[df_patient[path].iloc[i] for path in self.feature_list] for i in range(df_patient.shape[0])]
                data['glomeruli'].x = x

                # Create the target labels in tensor
                if self.onehot_targets:
                    df_patient = pd.get_dummies(df_patient, columns=['Term'])
                    # Add missing target columns if not represented in the data
                    for target in data.target_labels:
                        if target not in df_patient.columns:
                            df_patient[target] = False
                    y = df_patient[data.target_labels]
                    y = torch.tensor(y.to_numpy(), dtype=torch.float)
                else:
                    y = df_patient['Term']
                    y.replace({'Healthy': 0, 'Sclerotic': 1, 'Dead': 2}, inplace=True)
                    y = torch.tensor(y.to_numpy(), dtype=torch.long)
                data.y = y

                data.glom_indices = torch.tensor(df_patient['glom_index'].values)
                data.coords = torch.tensor(coords, dtype=torch.float)
                data.adjacency_matrix = torch.tensor(adjectency_matrix, dtype=torch.float)

                # Generate stratified train, val and test indices
                # Test set becomes equal to val set, if test split is 0
                # Val set becomes equal to train set, if val split is 0
                if (self.test_split < 1.0 and self.test_split > 0.0):
                    train_indices, test_indices = train_test_split(np.arange(len(y)),
                                                                   test_size=float(self.test_split),
                                                                   random_state=self.random_seed,
                                                                   stratify=y.numpy())
                    # Add a correction, because 100% for the val split will be reduced by the test split
                    val_split_correction = self.test_split * self.val_split
                else:
                    train_indices = np.arange(len(y))
                    test_indices = np.arange(len(y)) if patient in self.test_patients else np.array([])
                    val_split_correction = 0
                if (self.val_split < 1.0 and self.val_split > 0.0):
                    train_indices, val_indices = train_test_split(train_indices,
                                                                  test_size=float(
                                                                      self.val_split + val_split_correction),
                                                                  random_state=self.random_seed,
                                                                  stratify=y[train_indices].numpy())
                else:
                    train_indices = np.arange(len(y))
                    val_indices = np.arange(len(y)) if patient in self.val_patients else np.array([])

                # Add train, val and test masks based on random seed and split values
                data.train_mask = torch.zeros(data['glomeruli'].num_nodes, dtype=torch.bool)
                data.train_mask[train_indices] = True
                data.val_mask = torch.zeros(data['glomeruli'].num_nodes, dtype=torch.bool)
                data.val_mask[val_indices] = True
                data.test_mask = torch.zeros(data['glomeruli'].num_nodes, dtype=torch.bool)
                data.test_mask[test_indices] = True
                data.test_mask = data.val_mask if (self.test_split == 0.0) and (
                        self.test_patients == []) else data.test_mask

                # Preprocessing after determining train mask
                if self.path_image_inputs is None:
                    x = pd.DataFrame(data['glomeruli'].x.numpy())
                    x = feature_preprocessing(x, train_indices, **self.preprocessing_params)
                    data['glomeruli'].x = torch.tensor(x.to_numpy(), dtype=torch.float)

                # Add other cells to graph
                for i, df_cell_nodes in enumerate(list_cell_nodes):
                    # Create connections between cells
                    cell_coords = df_cell_nodes[['center_x_global', 'center_y_global']].to_numpy()
                    cell_sparse_matrix = knn_weighted_graph_construction(cell_coords, self.n_neighbours)

                    cell_edge_index = torch.tensor(cell_sparse_matrix.nonzero(), dtype=torch.long)
                    cell_edge_weight = torch.tensor(cell_sparse_matrix.data, dtype=torch.float).unsqueeze(1)
                    data[self.cell_types[i], 'to', self.cell_types[i]].edge_index = cell_edge_index
                    data[self.cell_types[i], 'to', self.cell_types[i]].edge_attr = cell_edge_weight

                    # Create connections between cells and glomeruli
                    df_glom_connection = df_patient['glom_index'].to_frame()
                    df_glom_connection['glom_row'] = np.arange(df_glom_connection.shape[0])
                    df_glom_connection = df_cell_nodes.merge(df_glom_connection, right_on='glom_row',
                                                             left_on='associated_glom', how='inner')
                    cell_glom_edge_index = (df_glom_connection.index.values, df_glom_connection['glom_row'])

                    data[self.cell_types[i], 'to', 'glomeruli'].edge_index = torch.tensor(cell_glom_edge_index,
                                                                                          dtype=torch.long)

                    # Get node features
                    x = df_cell_nodes[[c for c in df_cell_nodes.columns if c.endswith('_node_feature')]]
                    x = feature_preprocessing(x, list(range(0, len(x))), **self.preprocessing_params)
                    x = torch.tensor(x.to_numpy(), dtype=torch.float)

                    # Save cell data in graph object
                    data[self.cell_types[i]].x = x

                # Save graph data object
                file_name = f"{self.processed_file_name}_p{patient}.pt"
                torch.save(data, os.path.join(self.processed_dir, file_name))
                print(f'[Dataset]: Saves {file_name}')

                # Save file names of processed files and if patient is in train/test set and save in settings dict
                set = 'train' if patient in self.train_patients else 'validation' if patient in self.validation_patients else 'test'
                file_names.append({"file_name": file_name, "set": set})

        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'wb') as handle:
            pickle.dump(file_names, handle)

    def get(self, idx):
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images, if image features are used and transform to tensor
        if not isinstance(item['glomeruli'].x[0], torch.Tensor):
            images = []
            for paths in item.x:
                slices=[]
                for slice in paths:
                    slices.append(cv2.imread(slice))
                img = np.concatenate(slices, axis=2)
                images.append(img)
            item.x = np.array(images)
            item.x = torch.tensor(item.x, dtype=torch.float)
            item.x = item.x.permute(0, 3, 1, 2)

        return item
