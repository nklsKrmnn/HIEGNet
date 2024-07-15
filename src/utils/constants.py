from typing import Final
from itertools import product

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

TRAIN_PARAMETER_SEARCH_SPACE: Final[dict[str, list]] = {
    "learning_rate": [0.003, 0.001, 0.0003, 0.0001]
}

MODEL_PARAMETER_SEARCH_SPACE: Final[dict[str, list]] = {
    'hidden_dim': [16, 32, 64, 128],
    'message_passing_staps': [1, 2, 3, 5, 10],
    'n_fc_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'softmax_function': ['softmax', 'log_softmax', 'none'],
    'msg_passing_types': [{k: t[i] for i, k in enumerate(["glom_to_glom", "cell_to_glom", "cell_to_cell"])} for t in
                          list(product(*[['gat_v2', 'gcn', 'gin'] for i in range(3)]))],
    'norm_fc_layers': ['batch', 'layer', 'none'],
    'norm': ['batch', 'layer', 'none']
}
