from typing import Final

from src.models.gnn_cnn_hyrbid import GnnCnnHybrid
from src.models.gnn_hetero import HeteroGATv2
from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.pretrained_cnn import initialize_resnet
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.models.mlp import MLP
from src.preprocessing.datasets.image_dataset import GlomImageDataset
from src.preprocessing.datasets.hetero_graph_dataset import HeteroGraphDataset

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "gcn": GCN,
    "hybrid": GnnCnnHybrid,
    "mlp": MLP,
    "gcn_jk": GCNJumpingKnowledge,
    "gat_v2": GATv2,
    "resnet": initialize_resnet,
    "hetero_gat_v2": HeteroGATv2
}

DATASET_NAME_MAPPING: Final[dict[str, any]] = {
    "glom_graph_dataset": GlomGraphDataset,
    "image_dataset": GlomImageDataset,
    "hetero_graph_dataset": HeteroGraphDataset
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

