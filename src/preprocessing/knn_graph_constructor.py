import numpy as np

def knn_graph_constructor(X: np.array, k: int) -> np.array:
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
        nearest_neighbors = np.argsort(dists)[1:k+1]
        # Write into the adjacency matrix
        A[i, nearest_neighbors] = 1
        A[nearest_neighbors, i] = 1
    return A

if __name__ == "__main__":
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [2, 1]])
    k = 2
    A = knn_graph_constructor(X, k)
    print(A)