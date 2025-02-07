#!/bin/bash
# Export Python path
export PYTHONPATH=$(pwd)

# 1. Create glomeruli nodes

# Extract image glomeruli image patches and run stain deconvolution on image patches
python src/wsi_preprocessing/scripts/extract_glom_patches.py

# Create patches with black background
python src/wsi_preprocessing/scripts/mask_glom.py

# Extract glomeruli features
python src/wsi_preprocessing/scripts/extract_glom_image_features.py

# 2. Create tcell nodes

# Segment tcell nodes with cellpose
python src/wsi_preprocessing/scripts/cell_detection_tcell.py

# Create central glomeruli masks for cell feature within or outside of glomerulus
python src/wsi_preprocessing/scripts/create_central_glom_masks.py

# Extract tcell features
python src/wsi_preprocessing/scripts/cell_features_extraction.py -t tcell

# Create nodes
python src/wsi_preprocessing/scripts/cells2nodes.py

# Remove duplicates from overlapping glomeruli image patches
python src/wsi_preprocessing/scripts/cell_join.py -t tcell

# 3. Create macrophage nodes

# Segment tcell nodes with cellpose
python src/wsi_preprocessing/scripts/cell_detection_macro.py

# Extract tcell features
python src/wsi_preprocessing/scripts/cell_features_extraction.py -t M0

# Create nodes
python src/wsi_preprocessing/scripts/cells2nodes-macro.py

# Remove duplicates from overlapping glomeruli image patches
python src/wsi_preprocessing/scripts/cell_join.py -t M0
