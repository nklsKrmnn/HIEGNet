from typing import Final
from torchvision import models

from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GINEConv, RGCNConv, SAGEConv

from models.msg_passing_edge_sage import SAGEConvEdge
from src.models.msg_passing_schnet import SchNetMessagePassing

MESSAGE_PASSING_MAPPING: Final[dict[str, any]] = {
    "gcn": GCNConv,
    "gat_v2": GATv2Conv,
    "gin": GINConv,
    "gine": GINEConv,
    "rgcn": RGCNConv,
    'sage': SAGEConv,
    'cfconv': SchNetMessagePassing,
    'e_sage': SAGEConvEdge
}

RESNET_MODEL_MAPPING: Final[dict[int, any]] = {
    152: models.resnet152,
    101: models.resnet101,
    50: models.resnet50,
    34: models.resnet34,
    18: models.resnet18
}

EFFICIENT_NET_MAPPING: Final[dict[str, any]] = {
    's': models.efficientnet_v2_s,
    'm': models.efficientnet_v2_m,
    'l': models.efficientnet_v2_l
}
