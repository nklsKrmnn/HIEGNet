import numpy as np
import pandas as pd
import torch

MAX_GLOM_CELL_DISTANCE: float = 2200.0

def drop_cell_glom_edges(cell_data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all edges between cells and other glomeruli if the cell is inside a glomerulus.

    Args:
        cell_data (pd.DataFrame): DataFrame containing cell node data.

    Returns:
        pd.DataFrame: Same DataFrame with some glomeruli associations dropped.
    """
    # Select rows where the cell is not in a glomerulus
    in_glom_cells = cell_data[cell_data['is_in_glom']]

    # Drop other associated glomeruli
    in_glom_cells['associated_glomeruli'] = in_glom_cells['associated_glomeruli'].apply(
        lambda l: [d for d in l if d['is_in_glom']])

    return cell_data


def create_cell_glom_edges(df_cell_nodes: pd.DataFrame,
                           glom_index_series: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create edges between cells and glomeruli.

    Takes a DataFrame of cells as nodes and a Series of glomeruli indices. For each node in the DataFrame, it creates
    an edge to the glomerulus it is associated with. The edge feature is the distance between the cell and the
    glomerulus normalized and inverted, so it can be used as weighting (closer cells are closer to 1).

    :param df_cell_nodes: DataFrame with cell nodes.
    :param glom_index_series: Series with glomeruli indices.
    :return: Tuple of two tensors, one for the edge index and one for the edge distances.
    """
    # Isert reference with glomeruli indices
    df_cell_nodes['cell_row'] = np.arange(df_cell_nodes.shape[0])

    # Drop edges between cells and glomeruli if the cell is inside a glomerulus
    df_cell_nodes = drop_cell_glom_edges(df_cell_nodes)

    # Explode associated glomeruli to extract multiple edges
    df_cells_exploded = df_cell_nodes.explode('associated_glomeruli')
    df_cells_exploded['glom_index'] = df_cells_exploded['associated_glomeruli'].apply(lambda d: d['glom_index'])
    df_cells_exploded['distance'] = df_cells_exploded['associated_glomeruli'].apply(lambda d: d['distance'])
    df_glom_connection = glom_index_series.to_frame()
    df_glom_connection['glom_row'] = np.arange(df_glom_connection.shape[0])
    df_glom_connection = df_cells_exploded.merge(df_glom_connection, right_on='glom_index',
                                                 left_on='glom_index', how='inner')


    # Norm and invert distances to get weights
    #df_glom_connection['distance'] = 1 - df_glom_connection['distance'] / df_glom_connection['distance'].max()

    # Drop edges with distance greater than MAX_GLOM_CELL_DISTANCE
    df_glom_connection = df_glom_connection[df_glom_connection['distance'] <= MAX_GLOM_CELL_DISTANCE]

    # Create edge index
    cell_glom_edge_index = (df_glom_connection['cell_row'], df_glom_connection['glom_row'])

    # Create tensor for edge distances and index
    cell_glom_edge_distances = torch.tensor(df_glom_connection['distance'].values, dtype=torch.float).unsqueeze(1)
    cell_glom_edge_index = torch.tensor(df_glom_connection[['cell_row', 'glom_row']].values, dtype=torch.long).T

    return cell_glom_edge_index, cell_glom_edge_distances
