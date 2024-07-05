import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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

    # Use NearestNeighbors to find the k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
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


if __name__ == "__main__":
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [2, 1]])
    k = 2
    A = knn_graph_construction(X, k)
    print(A)
