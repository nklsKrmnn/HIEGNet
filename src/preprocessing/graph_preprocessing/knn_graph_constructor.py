import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def knn_graph_construction(X: np.array, k: int) -> np.array:
    """
    Construct a k-nearest neighbor graph from the data points in X.

    Parameters
    ----------
    X : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    k : int
        The number of nearest neighbors to consider for each data point.

    Returns
    -------
    numpy array
        A numpy array of shape (n, n) representing the adjacency matrix of the k-nearest neighbor graph.
    """
    n = X.shape[0]
    A = np.zeros((n, n))
    # iterate over each data point
    for i in range(n):
        # calculate the Euclidean distance between the current data point and all other data points
        dists = np.linalg.norm(X - X[i], axis=1)
        # Get knn (ignore first, because it is the data point itself)
        nearest_neighbors = np.argsort(dists)[1:k + 1]
        # Write into the adjacency matrix
        A[i, nearest_neighbors] = 1
        A[nearest_neighbors, i] = 1
    return A


def knn_weighted_graph_construction(X: np.array, k: int) -> csr_matrix:
    """
    Construct a k-nearest neighbor graph from the data points in X and add a weight to each edge based on the
    distance. The weights are normalized to the range [0, 1].

    Parameters
    ----------
    X : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    k : int
        The number of nearest neighbors to consider for each data point.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (n, n) representing the adjacency matrix of the k-nearest neighbor graph.
    """
    n = X.shape[0]

    if k == 0:
        return csr_matrix((n, n))
    else:
        # Use NearestNeighbors to find the k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Remove the first column (distances to self, which are zero)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Normalize the distances to the range [0, 1]
        max_distance = np.max(distances)
        distances = distances / max_distance if max_distance > 0 else distances

        # Create the sparse matrix
        row_indices = np.repeat(np.arange(n), k)
        col_indices = indices.flatten()
        data = distances.flatten()

        A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

        # Make the matrix symmetric by adding its transpose and then normalizing again
        A = A + A.T
        A.data = A.data / np.max(A.data)

        return A



import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix


def delaunay_graph_construction(X: np.array) -> csr_matrix:
    """
    Construct a graph from the data points in X using Delaunay triangulation.
    Add a weight to each edge based on the distance, normalized to the range [0, 1].

    Parameters
    ----------
    X : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (n, n) representing the adjacency matrix of the graph.
    """
    n = X.shape[0]

    # Perform Delaunay triangulation
    tri = Delaunay(X)

    # Extract edges
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.add((simplex[i], simplex[j]))

    # Create lists for row indices, column indices, and data (distances)
    row_indices = []
    col_indices = []
    data = []

    for (i, j) in edges:
        dist = np.linalg.norm(X[i] - X[j])
        row_indices.append(i)
        col_indices.append(j)
        data.append(dist)
        row_indices.append(j)
        col_indices.append(i)
        data.append(dist)

    # Normalize the distances to the range [0, 1]
    max_distance = np.max(data)
    data = np.array(data) / max_distance if max_distance > 0 else np.array(data)

    # Invert the distances to get weights
    data = 1 - data

    # Create the sparse matrix
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    return A

import numpy as np
from scipy.sparse import csr_matrix

def radius_based_graph_construction(X: np.array, radius: float) -> csr_matrix:
    """
    Construct a graph from the data points in X using radius-based connectivity.
    Add a weight to each edge based on the distance, normalized to the range [0, 1].

    Parameters
    ----------
    X : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    radius : float
        The radius within which to connect data points.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (n, n) representing the adjacency matrix of the graph.
    """
    # Compute the radius neighbors graph
    A = radius_neighbors_graph(X, radius, mode='distance', include_self=False)

    # Normalize the distances to the range [0, 1]
    if A.data.size > 0:
        max_distance = A.data.max()
        A.data = A.data / max_distance if max_distance > 0 else A.data

        # Invert the distances to get weights
        A.data = 1 - A.data

    return A

def graph_construction(X: np.array, method: str, **kwargs) -> csr_matrix:
    """
    Construct a graph from the data points in X using the specified method.

    Parameters
    ----------
    X : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    method : str
        The method to use for constructing the graph. Possible values are 'knn', 'delaunay', and 'radius'.
    **kwargs
        Additional keyword arguments to pass to the graph construction function.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (n, n) representing the adjacency matrix of the graph.
    """
    if method == 'knn':
        sparse_matrix = knn_weighted_graph_construction(X, **kwargs)
    elif method == 'delaunay':
        sparse_matrix = delaunay_graph_construction(X)
    elif method == 'radius':
        sparse_matrix = radius_based_graph_construction(X, **kwargs)
    else:
        raise ValueError(f"Unknown graph construction method: {method}")

    edge_index = sparse_matrix.nonzero() #np.argwhere(sparse_matrix > 1).T
    edge_weights = sparse_matrix.data
    return edge_index, edge_weights


def plot_graph(X, A, title):
    G = nx.from_scipy_sparse_array(A)
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color='blue', node_size=50, edge_color='gray')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Visualization function
    data = pd.read_pickle('/home/dascim/data/3_extracted_features/EXC/cell_nodes/M0_cell_nodes.pkl')
    data = data[data['associated_glom'] == 1332369]

    # Generate random data points
    np.random.seed(42)
    points = np.random.rand(30, 2)
    points = data[['center_x_global', 'center_y_global']].to_numpy()

    # Construct graphs
    delaunay_graph = delaunay_graph_construction(points)
    radius_graph = radius_based_graph_construction(points, radius=100)
    knn_graph = knn_weighted_graph_construction(points, k=5)

    # Plot the graphs
    plot_graph(points, delaunay_graph, "Delaunay Triangulation Graph")
    plot_graph(points, radius_graph, "Radius-based Connectivity Graph (radius=0.3)")
    plot_graph(points, knn_graph, "3-Nearest Neighbors Graph")