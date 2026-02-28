#! /usr/bin/env bash

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")
URL=https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install torch-scatter torch-cluster \
	  -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install --no-cache-dir torch-spline-conv -f $URL;
pip install torch-sparse==0.6.13 \
	  -f https://data.pyg.org/whl/torch-1.11.0+cu113.html;
pip install torch-geometric==2.0.4
pip install wandb numba hdf5plugin plotly matplotlib pycocotools opencv-python scikit-video pandas ruamel.yaml
pip install "Pillow<10.0.0"
