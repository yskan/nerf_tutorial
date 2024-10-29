# Neural Radiance Fields

This repository contains a simple tutorial on how to implement a Neural Radiance Field (NeRF) in PyTorch.

## Prerequisites

To run the code, you need to have the following Python packages installed:
```
conda create -n nerf_tutorial python=3.10
conda activate nerf_tutorial
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118 # Check your CUDA version
pip install numpy==1.24.1 matplotlib ipympl imageio ipywidgets tqdm
```

## Usage
Open `nerf.ipynb` in Jupyter Notebook and run the cells.

## Explore further

Check out [Nerf Studio](https://nerf.studio/) for a more interactive experience.

Check out [LightGaussianSplatting](https://github.com/VITA-Group/LightGaussian) for a peek at gaussian splatting.