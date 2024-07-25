from typing import Final
from itertools import product

from src.models.cnn import CNN
from src.models.gnn_cnn_hyrbid import GnnCnnHybrid, HeteroHybridGNN
from src.models.gnn_hetero import HeteroGNN
from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.mlp import MLP
from src.models.pretrained_cnn import initialize_resnet
from src.preprocessing.datasets.glom_graph_dataset import GlomGraphDataset
from src.preprocessing.datasets.hetero_hybrid_graph_dataset import HeteroHybridGraphDataset
from src.preprocessing.datasets.hybrid_graph_dataset import HybridGraphDataset
from src.preprocessing.datasets.image_dataset import GlomImageDataset
from src.preprocessing.datasets.hetero_graph_dataset import HeteroGraphDataset

DATASET_NAME_MAPPING: Final[dict[str, any]] = {
    "glom_graph_dataset": GlomGraphDataset,
    "image_dataset": GlomImageDataset,
    "hetero_graph_dataset": HeteroGraphDataset,
    "hybrid_graph_dataset": HybridGraphDataset,
    "hetero_hybrid_graph_dataset": HeteroHybridGraphDataset
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
    'hidden_dim': [16, 32, 64],
    'n_message_passings': [1, 3, 5, 10],
    'n_fc_layers': [1, 2, 3],
    'dropout': [0.2, 0.4, 0.6, 0.8],
    'softmax_function': ['softmax', 'log_softmax', 'none'],
    'msg_passing_types': [{k: t[i] for i, k in enumerate(["glom_to_glom", "cell_to_glom", "cell_to_cell"])} for t in
                          list(product(*[['gat_v2', 'gcn', 'gin'] for i in range(3)]))],
    'norm_fc_layers': ['batch', 'layer', 'none'],
    'norm': ['batch', 'layer', 'none']
}
MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "gcn": GCN,
    "hybrid": GnnCnnHybrid,
    "mlp": MLP,
    "cnn": CNN,
    "gcn_jk": GCNJumpingKnowledge,
    "gat_v2": GATv2,
    "resnet": initialize_resnet,
    "hetero_gnn": HeteroGNN,
    "hetero_hybrid_gnn": HeteroHybridGNN
}
