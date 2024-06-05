from typing import Final

from src.models.gnn_cnn_hyrbid import GnnCnnHybrid
from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.test_model import TestModel
from src.preprocessing.graph_dataset import GraphDataset
from src.preprocessing.glom_graph_dataset import GlomGraphDataset
from src.models.mlp import MLP

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "test": TestModel,
    "gcn": GCN,
    "hybrid": GnnCnnHybrid,
    "mlp": MLP,
    "gcn_jk": GCNJumpingKnowledge,
    "gat_v2": GATv2
}

DATASET_NAME_MAPPING: Final[dict[str, any]] = {
    "graph_dataset": GraphDataset,
    "glom_graph_dataset": GlomGraphDataset
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

