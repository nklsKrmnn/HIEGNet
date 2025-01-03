import numpy as np
from typing_extensions import Union
import torch


def drop_edges(edge_index_dict: dict, edge_types:Union[str, list]) -> dict:
    """
    Drop edges from a graph based on edge type(s).

    :param edge_index_dict: dict
        Dictionary containing the edge index for each edge type.
    :param edge_types: Union[str, list]
        Edge type(s) to drop.
    :return: dict
        Dictionary containing the edge index for each edge type after dropping the specified edge type(s).
    """
    if isinstance(edge_types, str):
        edge_types = [edge_types]

    for edge_type in edge_types:
        et = eval(edge_type)
        edge_index_dict[et] = torch.tensor(np.zeros((2,0)), dtype=torch.int64)

    return edge_index_dict