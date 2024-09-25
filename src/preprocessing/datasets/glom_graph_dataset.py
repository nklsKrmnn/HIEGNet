import cv2
import pandas as pd
import numpy as np
import pickle
import torch
import os
from typing import Final

from sklearn.utils import compute_class_weight
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import StratifiedKFold

from preprocessing.preprocessing_constants import SCALER_OPTIONS
from src.preprocessing.datasets.dataset_utils.dataset_utils import list_annotation_file_names, \
    get_train_val_test_indices, create_mask
from src.preprocessing.feature_preprocessing import feature_preprocessing
from src.preprocessing.graph_preprocessing.knn_graph_constructor import graph_construction
from src.utils.path_io import get_path_up_to

ROOT_DIR: Final[str] = get_path_up_to(os.path.abspath(__file__), "repos")


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
                 glom_graph: dict,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 validation_patients: list[str] = [],
                 test_patients: list[str] = [],
                 split_action:str = 'load',
                 set_indices_path: str = "/repos/histograph/data/input/set_indices/test15_val15",
                 onehot_targets: bool = True,
                 preprocessing_params: dict = None,
                 transform=None,
                 pre_transform=None,
                 random_seed=None) -> None:
        """
        Initialize the dataset.

        :param root: Directory where the dataset and the graph objects should be stored.
        :param processed_file_name: File name of the processed graph file.
        :param feature_file_path: File path to the feature file.
        :param annotations_path: Path to the annotation directory.
        :param feature_list: List of features to use from feature file
        :param glom_graph: Dictionary with parameters for the graph construction
        :param validation_split: Validation split
        :param test_split: Test split
        :param train_patients: Patients to train on
        :param validation_patients: Patients of only validate on
        :param test_patients: Patients to only test on
        :param split_action: Action to perform on the set indices (load, save or split without save)
        :param set_indices_path: Path to folder with the set index files
        :param onehot_targets: Bool if to use onehot encoding for targets
        :param preprocessing_params: Parameters for preprocessing (especially normalisation)
        :param transform: not used
        :param pre_transform: not used
        :param random_seed: Random seed
        """

        self.processed_file_name = processed_file_name
        self.feature_file_path = ROOT_DIR + feature_file_path
        self.test_split = test_split if test_patients != [] else 0.0
        self.val_split = validation_split if validation_patients != [] else 0.0
        self.train_patients = train_patients
        self.validation_patients = validation_patients
        self.test_patients = test_patients
        self.split_action = split_action
        self.set_indices_path = ROOT_DIR + set_indices_path
        self.feature_list = feature_list
        self.glom_graph = glom_graph
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.annot_path = ROOT_DIR + str(annotations_path)
        self.preprocessing_params = preprocessing_params

        root = ROOT_DIR + root
        super(GlomGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """
        Get the raw file names of the dataset.

        First file is the features file. To that, all files from the annotation directory are appended.

        :return: List of raw file names
        """
        raw_files = [self.feature_file_path] + list_annotation_file_names(self.annot_path)
        return raw_files

    @property
    def input_file_paths(self) -> list[str]:
        """
        Get the list of paths to input files of the dataset.

        List of all paths to files with input data is read from file which stores all these file names.

        :return: List of input file names
        """
        return [self.feature_file_path]

    @property
    def annotation_file_paths(self) -> list[str]:
        """
        Get the list of paths to annotation files of the dataset.

        List of all paths to files with annotations is read from file which stores all these file names.

        :return: List of annotation file paths
        """
        return list_annotation_file_names(self.annot_path)

    @property
    def processed_file_names(self) -> list[str]:
        """
        Get the list of paths to processed files of the dataset.

        List of all paths to files with processed graph objects is read from file which stores all these file names.

        :return: List of processed file names
        """
        try:
            with open(os.path.join(self.processed_dir, f'{self.processed_file_name}_filenames.pkl'), 'rb') as handle:
                file_names = pickle.load(handle)
        except:
            file_names = []

        return [item['file_name'] for item in file_names]

    def _download(self):
        pass

    def process(self) -> None:
        """
        Process the raw data to the graph data format.

        The raw data of the features and the annotations is loaded from raw files and merged into one DataFrame. For
        each of the patients in the DataFrame the data is processed by calling a function to create a graph data
        object. Each of the processed graph objects is then stored as file in the root dir and the file names are
        saved in and extra file.

        :return: None
        """
        print('[Dataset]: Processing data')

        df = pd.read_csv(self.input_file_paths[0])
        data_objects = {}

        # Load additional feature dataframes (like image paths for full dataset)
        for path2 in self.input_file_paths[1:]:
            df2 = pd.read_csv(path2)
            # rop any column, that exists in both dataframes
            df2.drop(columns=[f for f in df2.columns if (f in df.columns) and f != 'glom_index'], inplace=True)
            # Iterate over the remaining dataframes
            df = pd.merge(df, df2, on='glom_index', how='inner')
        df.dropna(subset=[f for f in self.feature_list if f in df.columns], inplace=True)

        df_annotations = pd.concat([pd.read_csv(path) for path in self.annotation_file_paths])
        df = pd.merge(df, df_annotations, left_on="glom_index", right_on="ID", how="left")
        patients_in_raw_data = df['patient'].unique()

        file_names = []

        for patient in patients_in_raw_data:
            df_patient = df[df['patient'] == patient]

            # Drop rows where feature or image path is missing (most likely because no match through slices)
            df_patient.dropna(subset=self.feature_list, inplace=True)

            # threshold for minimum number of data points and check if patient is in train, val or test set
            if (df_patient.shape[0] > 10) and (
                    patient in self.train_patients + self.validation_patients + self.test_patients):
                # Create the data object for each graph
                data = self.create_graph_object(df_patient, patient)

                # Save graph data object
                file_name = f"{self.processed_file_name}_p{patient}.pt"
                data_objects.update({file_name: data})

                # Save file names of processed files and if patient is in train/test set and save in settings dict
                file_names.append({
                    "file_name": file_name,
                    "train_patient": patient in self.train_patients,
                    "validation_patient": patient in self.validation_patients,
                    "test_patient": patient in self.test_patients
                })

        # Scale glomeruli features in all graphs
        data_objects = self.scale_glomeruli(data_objects)

        # Save data objects
        for file_name, data in data_objects.items():
            torch.save(data, os.path.join(self.processed_dir, file_name))
            print(f'[Dataset]: Saves {file_name}')

        # Save target labels
        self.target_labels = data.target_labels

        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'wb') as handle:
            pickle.dump(file_names, handle)

    def create_graph_object(self, df_patient, patient: str) -> Data:
        """
        The raw data is processed to a graph data object with the features, target labels and edge index.

        First the graph ist constructed from the passed DataFrame which the graph construction method given in this
        dataset object. Afterward, the target labels are created from the data (Y) and the train, val, test split is
        applied. At last the feature tensor is generated from the input DataFrame and preprocessed based on the given
        preprocessing parameters and the train indices (for normalization).

        :param df_patient: DataFrame for one graph data object
        :return: Graph data object.
        """
        # Create the data object for each graph
        data = Data()

        # Target labels
        data.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']

        # Create the graph from point cloud and generate the edge index
        coords = df_patient[['Center X', 'Center Y']].to_numpy()
        edge_index, edge_weights = graph_construction(coords, **self.glom_graph)
        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

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
        x = self.create_features(df_patient, train_indices, self.feature_list, preprocessing_params={'scaler': None})

        data.x = x
        data.y = y

        data.glom_indices = torch.tensor(df_patient['glom_index'].values)
        data.coords = torch.tensor(coords, dtype=torch.float)

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
        all_glom_features = torch.cat([data.x for data in data_objects.values()], dim=0)
        all_train_masks = torch.cat([data.train_mask for data in data_objects.values()], dim=0)

        # Fit the scaler to the whole train set
        scaler = SCALER_OPTIONS[self.preprocessing_params['scaler']]()
        scaler.fit(all_glom_features[all_train_masks])

        # Scale the features for all graphs
        for data in data_objects.values():
            data.x = torch.tensor(scaler.transform(data.x), dtype=torch.float)

        return data_objects

    def create_targets(self, df_patient: pd.DataFrame, target_labels: list[str]) -> torch.Tensor:
        """
        Creates target tensor for patient.

        Creates a torch tensor storing the target values given with the target_labels list. Extracts the column "Term"
        from the input DataFrame and applies one-hot encoding or encodes the targets with ordinal numbers.
        :param df_patient: Input DataFrame
        :param target_labels: List of target labels
        :return: Target value tensor
        """
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
            y = torch.tensor(y.to_numpy(), dtype=torch.float)

        return y

    def create_features(self, df: pd.DataFrame,
                        train_indices: list[int],
                        feature_list: list[str],
                        preprocessing_params: dict) -> torch.Tensor:
        # Get numerical features
        x = feature_preprocessing(df, feature_list, train_indices, **preprocessing_params)
        return x

    @property
    def image_size(self):
        return None

    def get_set_indices(self) -> tuple[list, list, list]:
        """
        Get the indices of the train, validation and test graphs.
        :return: Tuple of lists with the indices of the train and test graphs
        """
        with open(os.path.join(self.processed_dir, f"{self.processed_file_name}_filenames.pkl"), 'rb') as handle:
            file_names = pickle.load(handle)
        train_indices = [i for i, file in enumerate(file_names) if file['train_patient']]
        validation_indices = [i for i, file in enumerate(file_names) if file['validation_patient']]
        test_indices = [i for i, file in enumerate(file_names) if file['test_patient']]
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
            if file["train_patient"]:
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
                      list(set(train_idx) | set(val_idx))]
        graphs = [torch.load(fp) for fp in file_paths]

        # Get y data for stratified kfold
        cols = graphs[0].target_labels if self.onehot_targets else ['term']
        y_data = []
        masks = []
        for i, graph in enumerate(graphs):
            # Get y data and graph index
            y_data.append(pd.DataFrame(graph.y.numpy(), columns=cols))
            if self.onehot_targets:
                y_data[i]["term"] = y_data[i].idxmax(axis=1)
            y_data[i]['graph'] = i

            # Get mask for train and val data
            mask = np.zeros(len(graph.y))
            mask[graph.train_mask + graph.val_mask] = 1
            masks.append(mask)

        # Concatenate y data and masks
        train_val_mask = np.concatenate(masks)
        y_data = pd.concat(y_data, axis=0)

        # Apply mask and keep old index to identify samples later in the whole graph
        train_val_data = y_data[train_val_mask == 1]
        train_val_data = train_val_data.reset_index()

        # Create StratifiedKFold object
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

        # Generate the train index masks for each fold based on all graphs
        for i, (train_index, _) in enumerate(skf.split(train_val_data.index, train_val_data['term'])):
            train_val_data[f"train_fold_{i}"] = False
            train_val_data.loc[train_index, f"train_fold_{i}"] = True

        # Apply the fold masks for each graph
        for idx, graph in enumerate(graphs):
            len_graph = len(graph.y)
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
