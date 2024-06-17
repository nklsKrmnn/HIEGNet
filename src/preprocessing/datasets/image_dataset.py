import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

from src.preprocessing.datasets.dataset_utils.dataset_utils import list_annotation_file_names, \
    list_neighborhood_image_paths
from src.utils.file_name_utils import get_glom_index


class GlomImageDataset(Dataset):
    def __init__(self,
                 annotations_path,
                 path_image_inputs,
                 random_seed: int = 42,
                 validation_split: float = 0.2,
                 test_split: float = 0.0,
                 train_patients: list[str] = [],
                 onehot_targets:bool=True):

        self.test_split = test_split
        self.val_split = validation_split
        self.train_patients = train_patients
        self.img_dir = path_image_inputs
        self.random_seed = random_seed
        self.onehot_targets = onehot_targets
        self.patient_settings = {"Patients": train_patients}

        annotation_paths = list_annotation_file_names(annotations_path)
        df_annotations = pd.concat([pd.read_csv(path) for path in annotation_paths])
        df_annotations['patient'] = df_annotations['Image filename'].apply(lambda x: int(x.split('_')[2]))
        df_annotations = df_annotations[df_annotations['patient'].isin(self.train_patients)]
        df_annotations = df_annotations[df_annotations['Term'] != 'Tissue'].reset_index(drop=True)

        self.targets = self.transform_targets(df_annotations)

        # Get image paths
        existing_indices = df_annotations['ID'].tolist()
        image_paths = []
        for patient in self.train_patients:
            image_paths += list_neighborhood_image_paths(patient, self.img_dir)
        image_paths = [(get_glom_index(im_p), im_p) for im_p in image_paths if get_glom_index(im_p) in existing_indices]

        # Sort paths by index
        image_paths.sort(key=lambda x: existing_indices.index(x[0]))
        image_paths = [s for _, s in image_paths]

        self.img_paths = image_paths

    @property
    def image_size(self):
        image = read_image(self.img_paths[0])
        return image.shape[0]

    def get_set_indices(self):
        # get indices of train and test patients
        validation_split = 1 - self.val_split
        test_split = self.test_split

        dataset_size = len(self)

        # random split
        indices = list(range(dataset_size))
        if self.test_split > 0:
            train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=self.random_seed,
                                                           stratify=self.targets)
        else:
            train_indices = indices
            test_indices = []
        val_split_correction = self.test_split * self.val_split
        train_indices, validation_indices = train_test_split(train_indices, test_size=validation_split - val_split_correction,
                                                             random_state=self.random_seed,
                                                             stratify=self.targets[train_indices])
        if test_indices == []:
            test_indices = validation_indices

        return train_indices, validation_indices, test_indices

    def get_class_weights(self) -> torch.Tensor:
        """
        Get the class weights for the dataset.

        The class weights are
        calculated based on the distribution of the target labels in the dataset.
        """
        # Labels of train data
        train_indices = self.get_set_indices()[0]
        y_labels = self.targets[train_indices]

        if self.onehot_targets:
            y_labels = y_labels.argmax(dim=1).numpy()
            label_classes = np.unique(y_labels)
        else:
            y_labels = y_labels.numpy()
            label_classes = np.unique(y_labels)

        class_weights = compute_class_weight('balanced',
                                             classes=label_classes,
                                             y=y_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        return class_weights_tensor

    def __len__(self):
        return len(self.img_paths)

    def transform_targets(self, df_annotations):
        # Target labels
        self.target_labels = ['Term_Healthy', 'Term_Sclerotic', 'Term_Dead']
        # Create the target labels in tensor
        if self.onehot_targets:
            df_annotations = pd.get_dummies(df_annotations, columns=['Term'])
            # Add missing target columns if not represented in the data
            for target in self.target_labels:
                if target not in df_annotations.columns:
                    df_annotations[target] = False
            y = df_annotations[self.target_labels]
            y = torch.tensor(y.to_numpy(), dtype=torch.float)
        else:
            y = df_annotations['Term']
            y.replace({'Healthy': 0, 'Sclerotic': 1, 'Dead': 2}, inplace=True)
            y = torch.tensor(y.to_numpy(), dtype=torch.long)
        return y

    def transform_image(self, image):
        return image / 255.0

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.transform_image(read_image(img_path))
        label = self.targets[idx]
        return image, label