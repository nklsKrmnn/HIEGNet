# HIEGNet: A Heterogenous Graph Neural Network Including the Immune Environment in Glomeruli Classification[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the official implementation of [*HIEGNet: A Heterogenous Graph Neural Network Including
the Immune Environment in Glomeruli Classification*](#TODO).

<!--
<div align=center>
%<img src=https://github.com/ChangminWu/ExpanderGNN/blob/public/img/illustration.jpg  width="50%">
%</div>
-->
## Requirements
An appropriate virtual environment can be created by:

1. Install the general requirements file
```
pip install -r requiemnts.txt
```

2. Install the torch requirements file
```
pip install -r requirements_torch.txt
``` 

If you intend to use your own svs files, you will need to install the `openslide-python` package. This is a python front end for a C library called `openslide`. to install the C library follow the instructions on the [openslide website](https://openslide.org/download/).

## Graph creation pipeline
To create the required node input datafrom WSI with glomeruli annotations, run the `pipeline.sh` file. With bash script start a series of scripts that we started manually during our experiments. We manually set paths in these scripts, that might need to be adjusted if you prefer another directory structure. We used a data directory structure as follows:
```
.
├── 1_cytomine_downloads
    └── EXC
        ├── annotations
            └── 25
        ├── roi
            └── 25
        └── svs_images
            └── 25
├── 2_images_preprocessed
    └── EXC
        ├── masks_glom_isolated
            └── 25
        ├── patches
            └── 25
        ├── patches_florsecent
            └── 25
        ├── patches_glom_isolated
            └── 25
        └── patches_masked
            └── 25
├── 3_extracted_features
    ├── cell_segmentation_models
    └── EXC
        ├── cell_nodes
        ├── masks_cellpose
            ├── M0
            └── tcell
├── 4_input_data_graphs
    ├── glom_graphs
    └── processed
└── 5_model_outputs
```


## Usage HIEGNet
To run a training set up a config file or select one from `./data/test_configs/` and run the following command:
```
python src/main.py --pipline train --trainer graph --config ./path/to/config.yaml
```

To run a training set up a config file or select one from `./data/test_configs/` and run the following command:
```
python src/main.py --pipline train --trainer graph --config ./path/to/config.yaml
```

## Contribution
#### Authors: 
+ Niklas Kormann
+ Masoud Ramuz
+ Zeeshan Nisar
+ Nadine S. Schaadt
+ Hendrik Annuth
+ Benjamin Doerr
+ Friedrich Feuerhake
+ Thomas Lampert
+ Johannes F. Lutzeyer
