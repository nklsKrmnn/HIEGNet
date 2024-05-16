import cv2
import pandas as pd
import numpy as np
import pickle
import torch
from PIL.Image import Image
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
import random

from src.preprocessing.knn_graph_constructor import knn_graph_constructor


class GraphDataset(Dataset):
    """
    Dataset class for the graph data.

    The dataset class is used to load the data from the raw file and process it to the graph data format with node
    features and target. Only usable for homogeneous graphs.

    Args:
        root (str): The root directory of the dataset.
        raw_file_name (str): The name of the raw file containing the data.
        test_split (float): The fraction of the data that is used for testing. Default is 0.2.
        random_seed (int, optional): The random seed for the train-test split. Default is None.
    """

    def __init__(self,
                 root,
                 raw_file_name,
                 feature_list: list,
                 test_split: float = 0.2,
                 onehot_targets: bool = True,
                 transform=None,
                 pre_transform=None,
                 random_seed=None,
                 path_image_inputs=None):

        self.raw_file_name = raw_file_name
        self.test_split = test_split
        self.feature_list = feature_list
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.path_image_inputs = path_image_inputs
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        return [self.raw_file_name]

    @property
    def processed_file_names(self) -> list[str]:
        try:
            with open(os.path.join(self.processed_dir, 'processed_filenames.pkl'), 'rb') as handle:
                file_names = pickle.load(handle)
        except:
            file_names = []

        return file_names

    def download(self):
        pass

    def process(self) -> None:
        """
        Process the raw data to the graph data format.

        The raw data is loaded from the raw file and a graph is constructed from the point cloud with the knn graph
        algorithm. Afterward, node fatures and note-wise targets for node classification are added. A train and test
        mask is added to the graph data object. Finally, the data object is saved to the processed directory.

        :return: None
        """
        df = pd.read_csv(self.raw_paths[0])
        patients = df['patient'].unique()

        file_names = []

        for patient in patients:
            df_patient = df[df['patient'] == patient]

            # threshold for minimum number of data points
            if df_patient.shape[0] > 10:

                # Create the graph from point cloud and generate the edge index
                coords = df_patient[['centroid_x', 'centroid_y']].to_numpy()
                adjectency_matrix = knn_graph_constructor(coords, 5)
                edge_index = torch.tensor(np.argwhere(adjectency_matrix == 1).T, dtype=torch.long)

                # Create the node features in tensor
                if self.path_image_inputs is None:
                    x = df_patient[self.feature_list]
                    x = x.to_numpy()
                    x = torch.tensor(x, dtype=torch.float)
                else:
                    existing_indices = df_patient['glom_matching_index'].tolist()
                    x = self.load_neighborhood_image_paths(patient)
                    x = np.array(x)[existing_indices]
                    image_size = cv2.imread(x[0]).shape[0]

                # Target labels
                target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

                # Create the target labels in tensor
                if self.onehot_targets:
                    df_patient = pd.get_dummies(df_patient, columns=['Term'])
                    # Add missing target columns if not represented in the data
                    for target in target_labels:
                        if target not in df_patient.columns:
                            df_patient[target] = False
                    y = df_patient[target_labels]
                    y = torch.tensor(y.to_numpy(), dtype=torch.float)
                else:
                    y = df_patient['Term']
                    y.replace({'Healthy': 0, 'Sclerotic': 1, 'Dead': 2}, inplace=True)
                    y = torch.tensor(y.to_numpy(), dtype=torch.long)

                # Create the data object for each graph
                data = Data(x=x, edge_index=edge_index, y=y)

                # Add random train and test masks to the data object #TODO: Does this make sense here?
                data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                random.seed(self.random_seed)
                train_indices = random.sample(range(data.num_nodes), int(data.num_nodes * (1 - self.test_split)))
                data.train_mask[train_indices] = True
                data.test_mask = ~data.train_mask
                data.target_labels = target_labels

                if self.path_image_inputs is not None:
                    data.image_size = image_size

                file_name = f"{self.raw_file_name.split('.')[0]}_{patient}.pt"

                torch.save(data, os.path.join(self.processed_dir, file_name))
                print(f'[Dataset]: Saves {file_name}')
                file_names.append(file_name)

        with open(os.path.join(self.processed_dir, 'processed_filenames.pkl'), 'wb') as handle:
            pickle.dump(file_names, handle)

    def load_neighborhood_image_paths(self, patient) -> list[str]:
        """
        Load the neighborhood images for a patient.

        The neighborhood images are loaded from the path_image_inputs directory and the images.

        :param patient: The patient id
        :return: List of paths of the neighborhood images for one graph
        """
        path = os.path.join(self.path_image_inputs, f"patient_00{patient}")
        image_paths = []
        for image in os.listdir(path):
            img = os.path.join(path, image)
            image_paths.append(img)

        return image_paths

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx):
        item = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

        # Load images, if image features are used and transform to tensor
        if not isinstance(item.x[0], torch.Tensor):
            images = []
            for image in item.x:
                img = cv2.imread(image)
                images.append(img)
            item.x = np.array(images)
            item.x = torch.tensor(item.x, dtype=torch.float)
            item.x = item.x.permute(0, 3, 1, 2)

        return item


class TestGraphDataset(Dataset):
    def __init__(self, root, raw_file_name, test_split: float = 0.2, transform=None, pre_transform=None):
        self.raw_file_name = raw_file_name
        self.test_split = test_split
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.raw_file_name]

    @property
    def processed_file_names(self):
        try:
            with open(os.path.join(self.processed_dir, 'processed_filenames.pkl'), 'rb') as handle:
                file_names = pickle.load(handle)
        except:
            file_names = []

        return file_names

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])

        patients = df['patient'].unique()
        stainings = df['staining'].unique()
        file_names = []

        i = 0

        for patient in patients:
            for staining in stainings:
                df_patient_staining = df[(df['patient'] == patient) & (df['staining'] == staining)]

                # threshold for minimum number of data points
                if df_patient_staining.shape[0] > 10:
                    # Create the graph from point cloud and generate the edge index
                    coords = df_patient_staining[['Center X', 'Center Y']].to_numpy()
                    adjectency_matrix = knn_graph_constructor(coords, 5)
                    edge_index = torch.tensor(np.argwhere(adjectency_matrix == 1).T, dtype=torch.long)

                    # Create the node features in tensor
                    x = df_patient_staining[['Center X', 'Center Y', 'patient', 'staining', 'Perimeter', 'Area']]
                    x = x.replace({True: 1, False: 0})
                    x = x.to_numpy()
                    x = torch.tensor(x, dtype=torch.float)

                    # Create the target labels in tensor
                    y = df_patient_staining[["Term_Dead", "Term_Healthy", "Term_Sclerotic"]] #TODO: Test onehot encoding
                    y = torch.tensor(y.to_numpy(), dtype=torch.long)

                    # Create the data object for each graph
                    data = Data(x=x, edge_index=edge_index, y=y)

                    # Add random train and test masks to the data object #TODO: Does this make sense here?
                    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                    train_indices = random.sample(range(data.num_nodes), int(data.num_nodes * (1-self.test_split)))
                    data.train_mask[train_indices] = True
                    data.test_mask = ~data.train_mask

                    file_name = f'{i}_p{patient}_s{staining}.pt'

                    torch.save(data, os.path.join(self.processed_dir, file_name))
                    print(f'Saves {file_name}')
                    file_names.append(file_name)

                    i += 1

        with open(os.path.join(self.processed_dir, 'processed_filenames.pkl'), 'wb') as handle:
            pickle.dump(file_names, handle)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))



if __name__ == "__main__":
    dataset = GraphDataset(root='/home/dascim/repos/histograph/data/input', raw_file_name='annotations_cleaned.csv')
    dataset.process()

    test = dataset[1]
    print(test)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    test = next(iter(dataloader))
    print(test)

    print('done')