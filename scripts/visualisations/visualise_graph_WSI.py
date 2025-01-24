import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fontTools.misc.bezierTools import epsilon
from networkx.classes import neighbors
import pandas as pd
from src.utils.path_io import get_glom_index, get_patient
import os
from src.preprocessing.graph_preprocessing.knn_graph_constructor import graph_construction, graph_connection
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors



IMAGE_NAME = "patch_p001_s25_i1330627.png" #"patch_p001_s25_i1333560.png"
STAINING = 25
patient = get_patient(IMAGE_NAME)
glom_id = get_glom_index(IMAGE_NAME)

annotation_path = f"/home/dascim/data/1_cytomine_downloads/EXC/annotations/25/annotations_00{patient}_25.csv"
df_annotations = pd.read_csv(annotation_path)

# Read image
image_path = f"/home/dascim/data/2_images_preprocessed/EXC/patches/25/{IMAGE_NAME}"
image = cv2.imread(image_path)
size = image.shape[0]
max_d = 2*size

# shade image
image = cv2.addWeighted(image, 0.5, np.ones_like(image)*255, 0.5, 0)

# Get coordinate of glomerulus
glom = df_annotations[df_annotations["ID"] == glom_id]
glom_center = (glom["Center X"].values[0], glom["Center Y"].values[0])

# Get close glom in image
close_gloms = df_annotations #[df_annotations["ID"] != glom_id]
close_gloms['offset_x'] = close_gloms['Center X'] - glom_center[0]
close_gloms['offset_y'] = close_gloms['Center Y'] - glom_center[1]
close_gloms = close_gloms[(close_gloms['offset_x']**2 + close_gloms['offset_y']**2) < max_d**2]
close_gloms.reset_index(inplace=True)

# Read glom masks
mask_dir = f"/home/dascim/data/2_images_preprocessed/EXC/masks_glom_isolated/25/"
central_mask = cv2.imread(f'{mask_dir}/{IMAGE_NAME}', cv2.IMREAD_GRAYSCALE)
neighbors_masks = []

for index, row in close_gloms.iterrows():
    mask_name = f"patch_p00{patient}_s25_i{row['ID']}.png"
    mask = cv2.imread(f'{mask_dir}/{mask_name}', cv2.IMREAD_GRAYSCALE)
    neighbors_masks.append(mask)

# Upscale masks
central_mask = cv2.resize(central_mask, (size, size))
neighbors_masks = [cv2.resize(mask, (size, size)) for mask in neighbors_masks]

color_mapping = {
    "Dead": (255, 0, 0),       # Red
    "Sclerotic": (255, 255, 0),  # Yellow
    "Healthy": (0, 255, 0),     # Green
}

glom_in_fig = []

# Visualize masks on the image with offsets applied
for mask, (_, row) in zip(neighbors_masks, close_gloms.iterrows()):
    offset_x = int(row['offset_x'])
    offset_y = int(-row['offset_y'])

    # Create an empty overlay with the same size as the image
    overlay = np.zeros_like(central_mask)

    # Calculate the position for placing the neighbor mask
    x_start = max(0, offset_x)
    y_start = max(0, offset_y)
    x_end = min(size, size + offset_x)
    y_end = min(size, size + offset_y)

    # Define where the mask should be pasted
    mask_x_start = max(0, -offset_x)
    mask_y_start = max(0, -offset_y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    # Check if the mask is completely outside the image frame
    if x_start >= size or y_start >= size or x_end <= 0 or y_end <= 0:
        continue  # Skip this mask as it's outside the image frame

    # Apply the mask as a transparent overlay
    mask_section = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    alpha = 0.3  # Transparency factor

    # Extract the region of the image corresponding to the mask
    image_section = image[y_start:y_end, x_start:x_end]

    term = row["Term"]  # Get the term for this glomerulus
    color = color_mapping[term]

    # Create a colored overlay for the mask
    color_overlay = np.zeros_like(image_section)
    color_overlay[..., 0] = color[2]  # Blue channel (OpenCV uses BGR)
    color_overlay[..., 1] = color[1]  # Green channel
    color_overlay[..., 2] = color[0]  # Red channel

    # Blend the original image and the colored overlay using the mask
    image[y_start:y_end, x_start:x_end] = (
            image_section * (1 - alpha * (mask_section / 255)[:, :, None]) +
            color_overlay * (alpha * (mask_section / 255)[:, :, None])
    ).astype(np.uint8)

    glom_in_fig.append(row["ID"])

###### Safe iamge with glomeruli masks ######
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig("1_glom_masks.png")


# Load immune cell masks
mask_dir = f"/home/dascim/data/3_extracted_features/EXC/masks_cellpose"
for _, row in close_gloms.iterrows():
    m0_masks = np.load(f'{mask_dir}/M0/M0_mask_p00{patient}_s{STAINING}_i{row["ID"]}.npy')
    tcell_masks = np.load(f'{mask_dir}/tcell/tcell_mask_p00{patient}_s{STAINING}_i{row["ID"]}.npy')
    m0_masks = (m0_masks > 0) * 255
    tcell_masks = (tcell_masks > 0) * 255

    offset_x = int(row['offset_x'])
    offset_y = int(-row['offset_y'])

    # Create an empty overlay with the same size as the image
    overlay = np.zeros_like(central_mask)

    # Calculate the position for placing the neighbor mask
    x_start = max(0, offset_x)
    y_start = max(0, offset_y)
    x_end = min(size, size + offset_x)
    y_end = min(size, size + offset_y)

    # Define where the mask should be pasted
    mask_x_start = max(0, -offset_x)
    mask_y_start = max(0, -offset_y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    # Check if the mask is completely outside the image frame
    if x_start >= size or y_start >= size or x_end <= 0 or y_end <= 0:
        continue  # Skip this mask as it's outside the image frame

    # Apply the mask as a transparent overlay
    m0_masks_section = m0_masks[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    tcell_masks_section = tcell_masks[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    alpha = 0.3  # Transparency factor

    # Extract the region of the image corresponding to the mask
    image_section = image[y_start:y_end, x_start:x_end]

    # Visualise masks on image
    m0_overlay = np.zeros_like(image_section)
    tcell_overlay = np.zeros_like(image_section)
    m0_overlay[..., 0] = 255
    tcell_overlay[..., 2] = 255
    tcell_overlay[..., 0] = 255
    alpha = 0.5

    image[y_start:y_end, x_start:x_end] = (
            image_section * (1 - alpha * (m0_masks_section / 255)[:, :, None]) +
            m0_overlay * (alpha * (m0_masks_section / 255)[:, :, None])
    ).astype(np.uint8)

    image[y_start:y_end, x_start:x_end] = (
            image_section * (1 - alpha * (tcell_masks_section / 255)[:, :, None]) +
            tcell_overlay * (alpha * (tcell_masks_section / 255)[:, :, None])
    ).astype(np.uint8)

###### Safe iamge with immune cell masks ######
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig("2_immune_masks.png")

# Get graph
cell_types = ["M0", "tcell"]
cell_graph_params = {
    "method": "knn",  # Options: "knn", "radius", "delaunay"
    "k": 5,
    "lim": 400
}

cell_node_dir_path = "/home/dascim/data/3_extracted_features/EXC/cell_nodes"
cell_node_files = [os.path.join(cell_node_dir_path, f"{cell_type}_cell_nodes.pkl") for
                                cell_type in cell_types]
list_cell_nodes = [pd.read_pickle(file) for file in cell_node_files]

edges = {}
nodes = {}
df_cell_nodes = list_cell_nodes[0]

# Add other cells to graph
tmp = []
for i, df_cell_nodes in enumerate(list_cell_nodes):
    # Gat all nodes for that patient
    df_cell_nodes = df_cell_nodes[df_cell_nodes['patient'] == patient].reset_index(drop=True)

    # Get all nodes in range of cnetral glom
    df_cell_nodes['center_x_local_1'] = df_cell_nodes['center_x_global'] - glom_center[0]
    df_cell_nodes['center_y_local_1'] = df_cell_nodes['center_y_global'] - glom_center[1]

    df_cell_nodes = df_cell_nodes[(df_cell_nodes['center_x_local_1'] >= -size*0.75) & (df_cell_nodes['center_x_local_1'] < size*0.75)]
    df_cell_nodes = df_cell_nodes[(df_cell_nodes['center_y_local_1'] >= -size*0.75) & (df_cell_nodes['center_y_local_1'] < size*0.75)]
    tmp.append(df_cell_nodes.reset_index(drop=True))

list_cell_nodes = tmp


# Add other cells to graph
for i, df_cell_nodes in enumerate(list_cell_nodes):
    current_cell_type = cell_types[i]


    # Create connections between cells
    edge_type = (current_cell_type, 'to', current_cell_type)
    cell_coords = df_cell_nodes[['center_x_local_1', 'center_y_local_1']].to_numpy()
    cell_edge_index, cell_edge_weight = graph_construction(cell_coords, **cell_graph_params)

    edges[edge_type] = (cell_edge_index, cell_edge_weight)
    nodes[current_cell_type] = cell_coords

    # Create connections to other cell types
    for j, df_other_cell_nodes in enumerate(list_cell_nodes):
        edge_type = (current_cell_type, 'to', cell_types[j])
        if i == j:
            # Skip same cell type
            continue

        df_other_cell_nodes = df_other_cell_nodes[df_other_cell_nodes['patient'] == patient].reset_index(
            drop=True)

        other_cell_coords = df_other_cell_nodes[['center_x_local_1', 'center_y_local_1']].to_numpy()
        other_cell_edge_index, other_cell_edge_weight = graph_connection(cell_coords, other_cell_coords,
                                                                         **cell_graph_params)

        edges[edge_type] = (other_cell_edge_index, other_cell_edge_weight)



# adjust local coordinates
nodes['M0'][:, 0] += size/2
nodes['M0'][:, 1] = size/2 - nodes['M0'][:, 1]
nodes['tcell'][:, 0] += size/2
nodes['tcell'][:, 1] = size/2 - nodes['tcell'][:, 1]

# Create graph
G = nx.Graph()
G.add_nodes_from(range(len(nodes['M0'])), cell_type='M0')
G.add_nodes_from(range(len(nodes['M0']), len(nodes['tcell']) + len(nodes['M0'])), cell_type='tcell')

# Add edges
for edge_type, (edge_index, edge_weight) in edges.items():
    for _, v in enumerate(edge_index[0]):
        u = edge_index[1][_]
        v = v + len(nodes['M0']) if edge_type[0] == 'tcell' else v
        u = u + len(nodes['M0']) if edge_type[2] == 'tcell' else u
        G.add_edge(v, u, edge_weight=edge_weight[i], edge_type=edge_type)

# Draw edges between cells
pos = {i: (node[0], node[1]) for i, node in enumerate(nodes['M0'])}
pos.update({i+len(nodes['M0']): (node[0], node[1]) for i, node in enumerate(nodes['tcell'])})

# Display the image with highlighted masks
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Add graph to image
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

# Set axis limits to match the image dimensions
height, width, _ = image.shape
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Invert y-axis for image coordinates

plt.axis('off')
print("Done")

###### Safe iamge with immune cell masks ######
plt.savefig("3_c2c.png", )

# Add glom nodes to graph
n_edges_c2c = len(nodes['M0'])+len(nodes['tcell'])
G.add_nodes_from(range(n_edges_c2c,n_edges_c2c + len(close_gloms)), cell_type='glom')
pos.update({_+n_edges_c2c: (row['offset_x']+size*0.5, -row['offset_y']+size*0.5) for _, row in close_gloms.iterrows()})

# Draw edges between cells and glomeruli
for ct, df_cell_nodes in enumerate(list_cell_nodes):
    current_cell_type = cell_types[ct]

    for v, cell in df_cell_nodes.iterrows():
        ass_ids = [item['glom_index'] for item in cell['associated_glomeruli']]
        v = v + len(nodes['M0']) if current_cell_type == 'tcell' else v
        for u, glom in close_gloms.iterrows():
            distance = [d['distance'] for d in cell['associated_glomeruli'] if d['glom_index'] == glom[('ID')]]
            edge_type = (current_cell_type, 'to', 'glom')
            if distance:
                u = u + n_edges_c2c
                edge_weight = distance[0]
                G.add_edge(v,u, edge_weight=edge_weight, edge_type=edge_type)

# Define edge types that should be black
black_edge_types = [("tcell","to","tcell"), ("M0","to","tcell"), ("tcell","to","M0") , ("M0","to","M0"), ('glom','to','glom')]

# Get the edge attributes
edge_types = nx.get_edge_attributes(G, 'edge_type')
edge_weights = nx.get_edge_attributes(G, 'edge_weight')

# Define color palettes for other edge types
type_colormaps = {
    ('tcell', 'to','glom'): cm.RdPu_r,
    ('M0', 'to', 'glom'): cm.Blues_r
}

# Prepare edge colors
edge_colors = []
for edge in G.edges:
    edge_type = edge_types[edge]
    if edge_type in black_edge_types:
        # Assign black color for specific edge types
        edge_colors.append('black')
    else:
        # Normalize the weight for the colormap
        cmap = type_colormaps[edge_type]
        norm = mcolors.Normalize(vmin=min(edge_weights.values()), vmax=max(edge_weights.values()))
        edge_colors.append(cmap(norm(edge_weights[edge])))

ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.3, ax=ax)

# Set axis limits to match the image dimensions
height, width, _ = image.shape
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Invert y-axis for image coordinates

plt.axis('off')
print("Done")

###### Safe iamge with immune cell masks ######
plt.savefig("4_c2g.png")

# Add glom2glom edges
epsilon = 550
G3 = nx.Graph()
G.add_nodes_from(range(len(close_gloms)), cell_type='glom')
pos.update({_: (row['offset_x']+size*0.5, -row['offset_y']+size*0.5) for _, row in close_gloms.iterrows()})

for u, glom in close_gloms.iterrows():
    for v, glom2 in close_gloms.iterrows():
        if u == v:
            continue
        distance = np.sqrt((glom['offset_x'] - glom2['offset_x'])**2 + (glom['offset_y'] - glom2['offset_y'])**2)
        if distance <= epsilon:
            edge_type = ('glom', 'to', 'glom')
            G3.add_edge(u,v, edge_weight=distance, edge_type=edge_type)

# Get the edge attributes
edge_types = nx.get_edge_attributes(G3, 'edge_type')
edge_weights = nx.get_edge_attributes(G3, 'edge_weight')


nx.draw_networkx_edges(G3, pos, edge_color='orange', alpha=1, ax=ax, width=2)

# Set axis limits to match the image dimensions
height, width, _ = image.shape
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Invert y-axis for image coordinates

plt.axis('off')
print("Done")

###### Safe iamge with immune cell masks ######
plt.savefig("5_g2g.png")
