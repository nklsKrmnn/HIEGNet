import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import matplotlib
import pandas as pd


#matplotlib.use('TkAgg')

def visualize_graph(coordinates, sparse_matrix, target_classes, predicted_classes, class_labels):
    """
    Visualizes a graph with nodes colored by their target class and outlined by their predicted class.

    Parameters:
        coordinates (list of tuples): List of (x, y) coordinates for the nodes.
        sparse_matrix (numpy array): Adjacency matrix of the graph.
        target_classes (list): Target classes for the nodes.
        predicted_classes (list): Predicted classes for the nodes.
    """
    # Create the graph from the adjacency matrix
    adj_matrix = edge_list_to_adjacency_matrix(sparse_matrix.T)
    G = nx.from_numpy_array(adj_matrix)

    # Define custom color map
    colors = ['green', 'yellow', 'red']
    target_color_map = {i: colors[i % len(colors)] for i in range(len(set(target_classes)))}
    predicted_color_map = {i: colors[i % len(colors)] for i in range(len(set(target_classes)))}

    # Create a dictionary for node positions
    pos = {i: coord for i, coord in enumerate(coordinates)}

    fig = plt.figure(figsize=(12, 8))

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw nodes with target class colors and predicted class outlines
    for node in G.nodes():
        target_class = target_classes[node]
        predicted_class = predicted_classes[node]

        node_color = target_color_map[target_class]
        edge_color = predicted_color_map[predicted_class]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color=[node_color],
            edgecolors=[edge_color],
            node_size=100,
            linewidths=2.5
        )

    # Draw node labels
    #nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', alpha=0.5)

    # Create legend for target classes
    target_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='lightgrey', markerfacecolor=colors[i % len(colors)], markersize=10)
        for i in range(len(set(target_classes)))]
    target_legend_labels = [f"Target: {class_labels[i]}" for i in range(len(set(target_classes)))]
    target_legend = plt.legend(handles=target_legend_handles, labels=target_legend_labels, loc='upper left')

    # Create legend for predicted classes
    predicted_legend_handles = [
        plt.Line2D([0], [0], marker='o', color=colors[i % len(colors)], markerfacecolor='lightgrey', markersize=10, markeredgewidth=2)
        for i in range(len(set(class_labels)))]
    predicted_legend_labels = [f"Prediction: {class_labels[i]}" for i in range(len(set(target_classes)))]
    predicted_legend = plt.legend(handles=predicted_legend_handles, labels=predicted_legend_labels, loc='upper right')

    # Add legends to the plot
    plt.gca().add_artist(target_legend)
    plt.gca().add_artist(predicted_legend)

    plt.axis('off')
    plt.show()

    plt.savefig('graph.png')

    return fig


def edge_list_to_adjacency_matrix(edge_list):
    """
    Convert an edge list to an adjacency matrix.

    Parameters:
        edge_list (numpy array): 2D array where each row represents an edge between two nodes.

    Returns:
        numpy array: Adjacency matrix of the graph.
    """
    # Get the maximum node index to determine the size of the adjacency matrix
    num_nodes = int(np.max(edge_list)) + 1

    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Populate the adjacency matrix
    for edge in edge_list:
        node1, node2 = edge
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Assuming an undirected graph

    return adjacency_matrix