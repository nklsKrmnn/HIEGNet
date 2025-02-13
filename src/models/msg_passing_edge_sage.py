from typing import Union

import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_geometric.utils import spmm


class SAGEConvEdge(SAGEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_edge = torch.nn.LazyLinear(kwargs['out_channels'])

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: Tensor,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
            edge_attr = self.lin_edge(edge_attr).relu()

        self.edge_attr_tmp = edge_attr

            # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return torch.cat([x_j, self.edge_attr_tmp], dim=1)
