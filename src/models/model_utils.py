from torch import nn

def init_norm_layer(norm: str):
    if norm == "batch":
        return nn.BatchNorm1d
    elif norm == "layer":
        return nn.LayerNorm
    else:
        return nn.Identity