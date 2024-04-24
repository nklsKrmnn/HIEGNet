import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
import random

from src.preprocessing.knn_graph_constructor import knn_graph_constructor



class GraphDataset(Dataset):
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