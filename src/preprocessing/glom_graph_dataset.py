import cv2
import pandas as pd
import numpy as np
import pickle
import torch
import os
import random

from sklearn.utils import compute_class_weight
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.preprocessing.feature_preprocessing import feature_preprocessing
from src.preprocessing.knn_graph_constructor import knn_graph_constructor
from src.utils.file_name_utils import get_glom_index


class GlomGraphDataset(Dataset):
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
                 processed_file_name: str,
                 feature_file_path: str,
                 annotations_path: str,
                 feature_list: list,
                 n_neighbours: int = 5,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 validation_patients: list[str] = [],
                 test_patients: list[str] = [],
                 onehot_targets: bool = True,
                 preprocessing_params: dict = None,
                 transform=None,
                 pre_transform=None,
                 random_seed=None,
                 path_image_inputs=None):

        self.processed_file_name = processed_file_name
        self.feature_file_path = feature_file_path
        self.test_split = test_split if test_patients == [] else 0.0
        self.val_split = validation_split if (validation_patients == [] or validation_split==1.0) else 0.0
        self.train_patients = train_patients
        self.validation_patients = validation_patients if validation_patients != [] else train_patients
        self.test_patients = test_patients if test_patients != [] else validation_patients
        self.feature_list = feature_list
        self.n_neighbours = n_neighbours
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.path_image_inputs = path_image_inputs
        self.annot_path = str(annotations_path)
        self.preprocessing_params = preprocessing_params

        # Dict to save settings of patients in config later
        self.patient_settings = {
            "0_train_from_config": self.train_patients,
            "1_val_from_config": self.validation_patients,
            "2_test_from_config": self.test_patients,
            "3_train_in_dataset": [],
            "4_val_in_dataset": [],
            "5_test_in_dataset": [],
            "6_not_in_config": [],
            "7_to_small": []
        }

        super(GlomGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        raw_files = [self.feature_file_path]
        for root, _, files in os.walk(self.annot_path):
            for file in files:
                if file.endswith('.csv'):
                    raw_files.append(os.path.join(root, file))

        return raw_files

    @property
    def processed_file_names(self) -> list[str]:
        try:
            with open(os.path.join(self.processed_dir, f'{self.processed_file_name}_filenames.pkl'), 'rb') as handle:
                file_names = pickle.load(handle)
        except:
            file_names = []

        return [item['file_name'] for item in file_names]

    def download(self):
        pass

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

        file_names = []

        for patient in patients_in_raw_data:
            df_patient = df[df['patient'] == patient]

            # threshold for minimum number of data points and check if patient is in train or test set
            if (df_patient.shape[0] > 10) and (patient in self.train_patients + self.validation_patients + self.test_patients):

                # Create the graph from point cloud and generate the edge index
                coords = df_patient[['Center X', 'Center Y']].to_numpy()
                adjectency_matrix = knn_graph_constructor(coords, self.n_neighbours)
                edge_index = torch.tensor(np.argwhere(adjectency_matrix == 1).T, dtype=torch.long)

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

                # Generate stratified train, val and test indices
                # Test set becomes equal to val set, if test split is 0
                # Val set becomes equal to train set, if val split is 0
                if (self.test_split < 1.0 and self.test_split > 0.0):
                    train_indices, test_indices = train_test_split(np.arange(len(y)),
                                                                   test_size=float(self.test_split),
                                                                   random_state=self.random_seed,
                                                                   stratify=y.numpy())
                    # Add a correction, because 100% for the val split will be reduced by the test split
                    val_split_correction = self.test_split*self.val_split
                else:
                    train_indices = np.arange(len(y))
                    test_indices = np.arange(len(y)) if patient in self.test_patients else np.array([])
                    val_split_correction = 0
                if (self.val_split < 1.0 and self.val_split > 0.0):
                    train_indices, val_indices = train_test_split(train_indices,
                                                                  test_size=float(self.val_split + val_split_correction),
                                                                  random_state=self.random_seed,
                                                                  stratify=y[train_indices].numpy())
                else:
                    train_indices = np.arange(len(y))
                    val_indices = np.arange(len(y)) if patient in self.val_patients else np.array([])

                # Create the node features in tensor
                if self.path_image_inputs is None:
                    # Get numerical features
                    x = df_patient[self.feature_list]
                    x = feature_preprocessing(x, train_indices, **self.preprocessing_params)
                    x = x.to_numpy()
                    x = torch.tensor(x, dtype=torch.float)
                else:
                    # Get image paths
                    existing_indices = df_patient['glom_index'].tolist()
                    x = self.load_neighborhood_image_paths(patient)
                    x = [(get_glom_index(im_p), im_p) for im_p in x if get_glom_index(im_p) in existing_indices]

                    # Sort paths by index
                    x.sort(key=lambda x: existing_indices.index(x[0]))
                    x = [s for _, s in x]

                    image_size = cv2.imread(x[0]).shape[0]

                # Create the data object for each graph
                data = Data(x=x, edge_index=edge_index, y=y)

                # Add train, val and test masks based on random seed and split values
                data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                data.train_mask[train_indices] = True
                data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                data.val_mask[val_indices] = True
                data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool) #~(data.train_mask | data.val_mask) if self.test_split > 0 else data.val_mask
                data.test_mask[test_indices] = True
                data.test_mask = data.val_mask if (self.test_split == 0.0) and (self.test_patients==[]) else data.test_mask

                data.target_labels = target_labels

                if self.path_image_inputs is not None:
                    data.image_size = image_size

                file_name = f"{self.processed_file_name}_p{patient}.pt"

                # Save graph data object
                torch.save(data, os.path.join(self.processed_dir, file_name))
                print(f'[Dataset]: Saves {file_name}')

                # Save file names of processed files and if patient is in train/test set and save in settings dict
                if patient in self.train_patients:
                    self.patient_settings["3_train_in_dataset"].append(patient)
                    set = 'train'
                elif patient in self.validation_patients:
                    self.patient_settings["4_val_in_dataset"].append(patient)
                    set = 'validation'
                else:
                    self.patient_settings["5_test_in_dataset"].append(patient)
                    set = 'test'
                file_names.append({"file_name": file_name, "set": set})

            elif df_patient.shape[0] < 10:
                self.patient_settings["7_to_small"].append(patient)
            else:
                self.patient_settings["6_not_in_config"].append(patient)

        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'wb') as handle:
            pickle.dump(file_names, handle)

    def get_set_indices(self) -> tuple[list[int], list[int]]:
        """
        Get the indices of the train, validation and test graphs.
        :return: Tuple of lists with the indices of the train and test graphs
        """
        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'rb') as handle:
            file_names = pickle.load(handle)
        train_indices = [i for i, file in enumerate(file_names) if file['set'] == 'train']
        validation_indices = [i for i, file in enumerate(file_names) if file['set'] == 'validation']
        test_indices = [i for i, file in enumerate(file_names) if file['set'] == 'test']
        return train_indices, validation_indices, test_indices

    def get_class_weights(self) -> torch.Tensor:
        """
        Get the class weights for the dataset.

        The class weights are
        calculated based on the distribution of the target labels in the dataset.
        """
        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'rb') as handle:
            file_names = pickle.load(handle)
        y = []
        for file in file_names:
            if file["set"] == 'train':
                data = torch.load(os.path.join(self.processed_dir, file['file_name']))
                y.append(data.y)
        y = torch.cat(y)
        if self.onehot_targets:
            y_labels = y.argmax(dim=1).numpy()
            label_classes = np.unique(y_labels)
        else:
            y_labels = y.numpy()
            label_classes = np.unique(y_labels)

        class_weights = compute_class_weight('balanced',
                                             classes=label_classes,
                                             y=y_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        return class_weights_tensor

    def load_neighborhood_image_paths(self, patient) -> list[str]:
        """
        Load the neighborhood images for a patient.

        The neighborhood images are loaded from the path_image_inputs directory and the images.

        :param patient: The patient id
        :return: List of paths of the neighborhood images for one graph
        """
        dir = os.path.join(self.path_image_inputs)
        image_paths = []
        for im_file_name in os.listdir(dir):
            img_path = os.path.join(dir, im_file_name)
            if f"p00{patient}" in img_path:
                image_paths.append(img_path)

        return image_paths

    def create_folds(self, n_folds: int) -> None:
        """
        Create the n folds for the dataset.

        Generates folds with equal distribution of the target labels for the train and validation dataset. For each graph a list of
        train and test masks is generated for each fold. Class labels are stratified over the whole dataset not on
        each graph. The random seed of the dataset determines the fold generation. The folds are saved into each
        graph data object and save under the same file name.

        :param n_folds: Number of folds
        """
        # Get train and val graphs
        train_idx, val_idx, _ = self.get_set_indices()
        file_paths = [os.path.join(self.processed_dir, self.processed_file_names[idx]) for idx in
                      train_idx + val_idx]
        graphs = [torch.load(fp) for fp in file_paths]

        # Get y data for stratified kfold
        cols = graphs[0].target_labels
        y_data = []
        masks = []
        for i, graph in enumerate(graphs):
            # Get y data and graph index
            y_data.append(pd.DataFrame(graph.y.numpy(), columns=cols))
            y_data[i]["term"] = y_data[i].idxmax(axis=1)
            y_data[i]['graph'] = i
            
            # Get mask for train and val data
            mask = np.zeros(len(graph.y))
            mask[graph.train_mask+graph.val_mask] = 1
            masks.append(mask)
            
        # Concatenate y data and masks
        train_val_mask = np.concatenate(masks)
        y_data = pd.concat(y_data, axis=0)
        
        # Apply mask and keep old index to identify samples later in the whole graph
        train_val_data = y_data[train_val_mask==1]
        train_val_data = train_val_data.reset_index()

        # Create StratifiedKFold object
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

        # Generate the train index masks for each fold based on all graphs
        for i, (train_index, _) in enumerate(skf.split(train_val_data.index, train_val_data['term'])):
            train_val_data[f"train_fold_{i}"] = False
            train_val_data.loc[train_index, f"train_fold_{i}"] = True

        # Apply the fold masks for each graph
        for idx, graph in enumerate(graphs):
            len_graph = graph.num_nodes
            graph.train_folds = []
            graph.val_folds = []
            zero_mask = torch.zeros(len_graph, dtype=torch.bool)

            for fold in range(n_folds):
                # Create train mask using graph number and old index
                train_mask = zero_mask.clone()
                train_rows = train_val_data.loc[(train_val_data['graph'] == idx) & train_val_data[f"train_fold_{fold}"]]
                train_mask[list(train_rows["index"])] = True

                # Create val mask using graph number and old index
                val_mask = zero_mask.clone()
                val_rows = train_val_data.loc[(train_val_data['graph'] == idx) & ~train_val_data[f"train_fold_{fold}"]]
                val_mask[list(val_rows["index"])] = True

                # Safe folds in graph data object
                graph.train_folds.append(train_mask)
                graph.val_folds.append(val_mask)

            # Save graph after adding the fold masks
            torch.save(graph, file_paths[idx])

    def activate_fold(self, fold: int) -> None:
        """
        Activate the fold for the dataset.

        Iterates over all graphs in the dataset and sets the train and test mask for the corresponding mask in the
        given fold.

        :param fold: The fold to activate
        """

        train_idx, val_idx, _ = self.get_set_indices()
        indices = train_idx + val_idx
        # Load graph data objects
        file_paths = [os.path.join(self.processed_dir, self.processed_file_names[idx]) for idx in
                      indices]
        graphs = [torch.load(fp) for fp in file_paths]

        for idx, graph in enumerate(graphs):
            # Set train and test mask for the given fold
            graph.train_mask = graph.train_folds[fold]
            graph.val_mask = graph.val_folds[fold]
            if self.test_patients == [] and self.test_split == 0.0:
                graph.test_mask = graph.val_mask

            # Save graph after adding the fold masks
            torch.save(graph, file_paths[idx])

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
