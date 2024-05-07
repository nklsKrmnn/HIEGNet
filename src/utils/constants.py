from typing import Final

from src.models.gnn_cnn_hyrbid import GnnCnnHybrid
from src.models.gnn_models import TestGNN
from src.models.test_model import TestModel
from src.preprocessing.graph_dataset import GraphDataset

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "test": TestModel,
    "test_gnn": TestGNN,
    "hybrid": GnnCnnHybrid
}

DATASET_NAME_MAPPING: Final[dict[str, any]] = {
    "graph_dataset": GraphDataset
}

REFERENCE_POINTS: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "001": {
        "origin": (16038.24, 69265.17),
        "target": (380, 1001)
    },
    "002": {
        "origin": (20212.06, 59714.98),
        "target": (632, 1261)
    }
}

