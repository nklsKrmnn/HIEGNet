#!/bin/bash

# Define the folder containing the config files
CONFIG_DIR="data/bash_configs"

# Export Python path
export PYTHONPATH=$(pwd)

# Loop through all YAML files in the given folder
for config in "$CONFIG_DIR"/*.yaml; do
  if [[ -f "$config" ]]; then
    echo "Running training with config: $config"
    python src/main.py --pipeline evaluate --trainer graph --config "$config"
  fi
done
#python src/main.py --pipeline evaluate --trainer graph --config data/eval_configs/eval_HIEGNet_between_config.yaml
#python src/main.py --pipeline evaluate --trainer graph --config data/eval_configs/eval_HIEGNet_within_config.yaml

