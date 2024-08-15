from typing import Final
from src.models.cnn import CNN
from src.models.full_model import FullHybrid
from src.models.gnn_cnn_hybrid_skip import SkipHeteroHybridGNN
from src.models.gnn_cnn_hyrbid import GnnCnnHybrid, HeteroHybridGNN
from src.models.gnn_hetero import HeteroGNN
from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.mlp import MLP
from src.models.pretrained_cnn import initialize_resnet
from src.preprocessing.datasets.full_graph import FullGraphDataset
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
    "hetero_hybrid_graph_dataset": HeteroHybridGraphDataset,
    "full_graph_dataset": FullGraphDataset
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

PARAMETER_SEARCH_SPACE: Final[dict[str, list]] = {
    'hidden_dim': [16, 32, 64],
    'n_message_passings': [1, 2, 3, 4, 5],
    'n_fc_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'softmax_function': ['softmax', 'log_softmax', 'none'],
    'norm_fc_layers': ['batch', 'layer', 'none'],
    'norm': ['batch', 'layer', 'none'],
    'glom_to_glom': ['gcn', 'gat_v2', 'gine'],
    'cell_to_glom': ['gcn', 'gat_v2', 'gine'],
    'cell_to_cell': ['gcn', 'gat_v2', 'gine'],
    "learning_rate": [0.003, 0.001, 0.0003, 0.0001]
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
    "hetero_hybrid_gnn": HeteroHybridGNN,
    "hetero_hybrid_gnn_skip": SkipHeteroHybridGNN,
    "hetero_full_model": FullHybrid
}
