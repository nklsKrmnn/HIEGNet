#!/bin/bash

# List of config files
configs=(
  "data/bash_configs/hetero_graph_unified_delauney.yaml"
  "data/bash_configs/hetero_graph_unified_knn5.yaml"
  "data/bash_configs/hetero_graph_knn5.yaml"
  # Add more config files as needed
)

export PYTHONPATH=$(pwd)
# Loop through each config file and run the command
for config in "${configs[@]}"; do
  echo "Running training with config: $config"
  python src/main.py --pipeline grid_search --trainer graph --config "$config"
done