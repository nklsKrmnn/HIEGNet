from typing import Final

from src.models.gnn_models import TestGNN
from src.models.test_model import TestModel
from src.preprocessing.test_graph_dataset import GraphDataset

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "test": TestModel,
    "test_gnn": TestGNN
}

DATASET_NAME_MAPPING: Final[dict[str, any]] = {
    "graph_dataset": GraphDataset
}

