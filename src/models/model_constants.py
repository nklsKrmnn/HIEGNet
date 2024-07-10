from typing import Final

from torch_geometric.nn import GCNConv, GATv2Conv, GINConv

MESSAGE_PASSING_MAPPING: Final[dict[str, any]] = {
    "gcn": GCNConv,
    "gat_v2": GATv2Conv,
    "gin": GINConv
}
