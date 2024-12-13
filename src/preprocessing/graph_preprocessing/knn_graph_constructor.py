import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.spatial import Delaunay
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


def knn_feature_graph_construction(X_1: np.array, X_2: np.array, k: int) -> csr_matrix:
    """
    Construct a k-nearest neighbor graph between two sets of data points X_1 and X_2 and add the distnace as edge
    features. To use one set of data points, set X_1 and X_2 to the same value.

    Parameters
    ----------
    X_1 : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    X_2 : numpy array
        A numpy array of shape (n, d) where n is the number of data points and d is the number of features.
    k : int
        The number of nearest neighbors to consider for each data point.

    Returns
    -------
    csr_matrix
        A sparse matrix of shape (n, n) representing the adjacency matrix of the k-nearest neighbor graph.
    """
    n_1 = X_1.shape[0]
    n_2 = X_2.shape[0]

    if k == 0:
        return csr_matrix((n_1, n_2))
    else:
        # Add 1 to k because the nearest neighbor is the data point itself (only within one set)
        if X_1 is X_2:
            k += 1
            first_nbr_idx = 1
        else:
            first_nbr_idx = 0

        # Use NearestNeighbors to find the k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_2)
        distances, indices = nbrs.kneighbors(X_1)

        # Remove the first column (distances to self, which are zero)
        distances = distances[:, first_nbr_idx:]
        indices = indices[:, first_nbr_idx:]

        # Normalize the distances to the range [0, 1]
        # max_distance = np.max(distances)
        # distances = distances / max_distance if max_distance > 0 else distances

        # Create the sparse matrix
        row_indices = np.repeat(np.arange(n_1), k - first_nbr_idx)
        col_indices = indices.flatten()
        data = distances.flatten()

        A = csr_matrix((data, (row_indices, col_indices)), shape=(n_1, n_2))

        # Make the matrix symmetric by adding its transpose if there is one set
        if X_1 is X_2:
            A = A + A.T
            # Normalize the weights to the range [0, 1] after adding transposed
            A.data = A.data / np.max(A.data)

        return A


def delaunay_graph_construction(X: np.array) -> csr_matrix:
    """
    Construct a graph from the data points in X using Delaunay triangulation.
    The distance between the nodes is added as edge feature.

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
    # max_distance = np.max(data)
    # data = np.array(data) / max_distance if max_distance > 0 else np.array(data)

    # Invert the distances to get weights
    # data = 1 - data

    # Create the sparse matrix
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    return A


def radius_based_graph_construction(X: np.array, radius: float) -> csr_matrix:
    """
    Construct a graph from the data points in X using radius-based connectivity.
    The distance between the nodes is added as edge feature.

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

    # if A.data.size > 0:
    #    # Normalize the distances to the range [0, 1]
    #    max_distance = A.data.max()
    #    A.data = A.data / max_distance if max_distance > 0 else A.data

    #    # Invert the distances to get weights
    #    A.data = 1 - A.data

    # Add a small value to the distances to avoid zero values
    # A.data[A.data == 0] = 1e-10

    return A

def graph_construction(X: np.array, method: str, **kwargs) -> tuple[np.array, np.array]:
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
        sparse_matrix = knn_feature_graph_construction(X, X, **kwargs)
    elif method == 'delaunay':
        sparse_matrix = delaunay_graph_construction(X)
    elif method == 'radius':
        sparse_matrix = radius_based_graph_construction(X, **kwargs)
    else:
        raise ValueError(f"Unknown graph construction method: {method}")

    edge_index = sparse_matrix.nonzero()
    edge_weights = sparse_matrix.data
    return edge_index, edge_weights


def graph_connection(X_1: np.array, X_2: np.array, method: str, **kwargs) -> tuple[np.array, np.array]:
    """
    Connect two sets of data points using the specified method.

    :param X_1: Data points of the first set
    :param X_2: Data points of second set
    :param method: Method to use
    :param kwargs: Additional keyword arguments for the specified method
    :return:
    """
    if method == 'knn':
        sparse_matrix = knn_feature_graph_construction(X_1, X_2, **kwargs)
    elif method == 'delaunay':
        raise ValueError("Delaunay triangulation is not supported for connecting two sets of data points.")
    elif method == 'radius':
        raise ValueError("Radius-based connectivity is not supported for connecting two sets of data points.")
    else:
        raise ValueError(f"Unknown graph construction method: {method}")

    edge_index = sparse_matrix.nonzero()
    edge_features = sparse_matrix.data
    return edge_index, edge_features


def plot_graph(X, A, title, node_class: list = None):
    G = nx.from_scipy_sparse_array(A)
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color='blue', node_size=50, edge_color='gray')

    # Annote with class
    ann_color = {"A": 'red', "B": 'green'}
    if node_class is not None:
        for i in range(len(X)):
            plt.text(X[i, 0], X[i, 1], str(node_class[i]), fontsize=12, color=ann_color[node_class[i]])

    plt.title(title)
    plt.show()

    print('stop')


if __name__ == "__main__":
    # Visualization function
    data = pd.read_pickle('/home/dascim/data/3_extracted_features/EXC/cell_nodes/M0_cell_nodes.pkl')
    data = data[data['associated_glom'] == 1332369]

    # Generate random data points
    np.random.seed(42)
    points_1 = np.random.rand(20, 2)
    points_2 = np.random.rand(30, 2)
    # points = data[['center_x_global', 'center_y_global']].to_numpy()

    # Construct graphs
    # delaunay_graph = delaunay_graph_construction(points_1)
    # radius_graph = radius_based_graph_construction(points_1, radius=100)
    knn_graph = knn_feature_graph_construction(points_1, points_1, k=5)

    # Connect two sets of data points
    # knn_graph = knn_weighted_graph_construction(points_1, points_2, k=5)

    # Plot the graphs
    # plot_graph(points_1, delaunay_graph, "Delaunay Triangulation Graph")
    # plot_graph(points_1, radius_graph, "Radius-based Connectivity Graph (radius=0.3)")
    plot_graph(points_1, knn_graph, "5-Nearest Neighbors Graph")

    # set_annotation = ["A"] * 20 + ["B"] * 30
    # upper_half = hstack([csr_matrix((A.shape[0], A.shape[0])), A])
    # lower_half = hstack([A.T, csr_matrix((A.shape[1], A.shape[1]))])
    # A = vstack([upper_half, lower_half])
    # plot_graph(np.vstack((points_1, points_2)), knn_graph, "3-Nearest Neighbors Graph", set_annotation)
