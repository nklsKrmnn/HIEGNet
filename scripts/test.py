import numpy as np
import time
import tracemalloc


def knn_weighted_graph_construction_numpy(X: np.array, k: int) -> tuple:
    n = X.shape[0]

    row_indices = []
    col_indices = []
    data = []

    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        nearest_neighbors = np.argsort(dists)[1:k + 1]
        row_indices.extend([i] * k)
        col_indices.extend(nearest_neighbors)
        data.extend(dists[nearest_neighbors])

    max_distance = np.max(data)
    data = np.array(data) / max_distance if max_distance > 0 else np.array(data)

    row_indices = np.concatenate([row_indices, col_indices])
    col_indices = np.concatenate([col_indices, row_indices[:len(row_indices) // 2]])
    data = np.concatenate([data, data[:len(data) // 2]])

    data = data / np.max(data)

    return row_indices, col_indices, data


# Example usage for timing and memory
X = np.random.rand(1000, 50)  # Adjust size as needed
k = 10

# Measure time and memory for numpy version
tracemalloc.start()
start_time = time.time()
row_indices, col_indices, data = knn_weighted_graph_construction_numpy(X, k)
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"NumPy version - Time: {end_time - start_time} seconds")
print(f"NumPy version - Memory usage: {peak / 1024 / 1024} MB")
