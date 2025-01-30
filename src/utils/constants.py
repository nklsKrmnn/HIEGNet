from typing import Final

from models.mlp import MLP_dumm
from models.pretrained_cnn import initialize_efficientnet_v2
from models.resnet import resnet_custom
from preprocessing.datasets.hetero_graph_dataset2 import HeteroUnifiedGraphDataset
from preprocessing.datasets.tabular_dataset import TabularDataset
from src.models.cnn import CNN, CBR
from src.models.full_model import FullHybrid, FullHybridJK
from src.models.gnn_cnn_hybrid_skip import SkipHeteroHybridGNN
from src.models.gnn_cnn_hyrbid import GnnCnnHybrid, HeteroHybridGNN
from src.models.gnn_hetero import HeteroGNN, HeteroGnnJK
from src.models.gnn_models import GCN, GCNJumpingKnowledge, GATv2
from src.models.mlp import MLP
from src.models.pretrained_cnn import initialize_resnet, initialize_efficientnet_v2
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
    "full_graph_dataset": FullGraphDataset,
    "tabular_dataset": TabularDataset,
    "hetero_unified_graph_dataset":HeteroUnifiedGraphDataset
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
    'hidden_dim': [64, 128],
    'n_readout_layers': [1, 2],
    'n_message_passings': [2,3],
    'n_fc_layers': [1, 2],
    'dropout':  [0.0, 0.05, 0.1, 0.15],#[0.1, 0.2, 0.3, 0.4, 0.5],
    'mlp_dropout': [0.2, 0.4, 0.6, 0.8],
    'softmax_function': ['softmax', 'log_softmax', 'none'],
    'norm_fc_layers': ['batch', 'layer', 'none'],
    'norm': ['batch', 'layer', 'none'],
    'glom_to_glom': ['gin','gcn','gat_v2'], #['sage', 'gat_v2', 'e_sage'],
    'cell_to_glom': ['sage', 'gat_v2', 'e_sage'],
    'cell_to_cell': ['gcn', 'sage', 'gat_v2', 'gine', 'cfconv', 'e_sage'],
    "msg_passing_types": ['sage', 'gat_v2', 'e_sage'],
    "learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
    "fc_learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
    "max_lr": [0.01, 0.001],
    "layers": [34, 18],
    "enet_size": ['s', 'm'],
    "optimizer": ['adam', 'sgd'],
    "mlp_hidden_dim": [2, 4, 8, 16, 32, 64, 128, 256],
    "mlp_hidden_layers": [1, 2, 4, 8],
    "weight_dacay": [1e-3, 1e-4, 1e-5, 1e-6]
}

MODEL_NAME_MAPPING: Final[dict[str, any]] = {
    "gcn": GCN,
    "hybrid": GnnCnnHybrid,
    "mlp": MLP,
    "cnn": CNN,
    "gcn_jk": GCNJumpingKnowledge,
    "gat_v2": GATv2,
    "resnet_pretrained": initialize_resnet,
    "resnet_custom": resnet_custom,
    "efficientnet_v2_pretrained": initialize_efficientnet_v2,
    "hetero_gnn": HeteroGNN,
    "hetero_hybrid_gnn": HeteroHybridGNN,
    "hetero_hybrid_gnn_skip": SkipHeteroHybridGNN,
    "hetero_full_model": FullHybrid,
    "hetero_full_jk_model": FullHybridJK,
    "hetero_jk": HeteroGnnJK,
    "cbr": CBR,
    'mlp_dump': MLP_dumm
}
