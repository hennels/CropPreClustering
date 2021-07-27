# CropPreClustering
A repository containing point clouds of crops and some pre-clustering algorithms to operate on them. Meant as companion data and code for the paper [Pre-Clustering Point Clouds of Crop Fields Using Scalable Methods](https://arxiv.org/abs/2107.10950).

This repository contains code for all the algorithms defined in the paper written in Python. While Python is not an extremly performant language, it does make development easy with readily installable libraries for 3D file reading, spatial data structures, and visualization.

## Depedancies
The code in this repository depends on:
 - Numpy
 - Scipy
 - Numba
 - Open3D

If your system does not have these it is easier to use the included [Poetry](https://python-poetry.org/) environment. After Poetry is installed, to get a working environment you should be able to simply run
```
poetry install
```

## Usage
All algorithms are defined in `croppreclustering/paperalgs.py` if you would like to inspect the source. Each algorithm has a corresponding script in the `scripts/` folder and should be easily runnable with a `-h` parameter to print usage information.

# Data
In the `data/` folder there are four point clouds of corn fields on which the algorithms can be run. All point clouds were manually transformed so that the up direction in the world frame is pointed in the *+z* direction of the point cloud.

# Examples
Below are a few examples showing script execution and their results.
## Non-random RAIN
```
poetry run python scripts/nonrandomRAIN.py data/DjiV4.ply 0.5
```
![Z Quickshift Result](images/nRAIN.gif)

## Z Quickshift
```
poetry run python scripts/zquickshift.py data/DjiV4.ply 1.0
```
![Z Quickshift Result](images/Zquickshift.gif)

## Ground Density Quickshift
```
poetry run python scripts/gdquickshift.py data/DjiV4.ply 1500 1.5
```
![GD Quickshift Result](images/GDquickshift.gif)

## Ground Density Quickshift++
```
poetry run python scripts/gdquickshiftpp.py data/DjiV4.ply 1200 0.4
```
![GD Quickshift++ Result](images/GDquickshift++.gif)

# Evaluation
The script `scripts/compare_clustering.py` is provided to evaluate clusterings relative to some ground truth. The script computes the optimal relationship between ground truth and predicted clusters and prints mean and median IoU statistics to the screen.
```
poetry run python scripts/compare_clustering.py data/0to9_1M/0to9_1M.ply data/0to9_1M/0to9_1M_GDqspp_1500_pt4.ply
```
