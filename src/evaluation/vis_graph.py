import matplotlib.pyplot as plt
import networkx as nx

import matplotlib
matplotlib.use('TkAgg')

def visualize_graph(coordinates, adjacency_matrix, target_classes, predicted_classes, class_labels):
    """
    Visualizes a graph with nodes colored by their target class and outlined by their predicted class.

    Parameters:
        coordinates (list of tuples): List of (x, y) coordinates for the nodes.
        adjacency_matrix (numpy array): Adjacency matrix of the graph.
        target_classes (list): Target classes for the nodes.
        predicted_classes (list): Predicted classes for the nodes.
    """
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Define custom color map
    colors = ['green', 'yellow', 'red']
    target_color_map = {i: colors[i % len(colors)] for i in range(len(set(target_classes)))}
    predicted_color_map = {i: colors[i % len(colors)] for i in range(len(set(predicted_classes)))}

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