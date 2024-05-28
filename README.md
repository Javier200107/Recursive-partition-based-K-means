# Recursive-partition-based-K-means

This is a Python implementation of the Recursive partition-based K-means algorithm (RPKM). The algorithm is based on the paper "Recursive partition-based clustering algorithm for large data sets" by S. Murthy and S.K. Pal. The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0950705116302027).

RPKM is a clustering algorithm that is based on the K-means algorithm. It recursively partitions the data set into smaller clusters until the number of data points in each cluster is less than a specified threshold. The algorithm is useful for clustering large data sets that cannot be clustered using traditional K-means algorithms.

## Run instructions

Create a new conda environment and install the required packages:

```bash
conda create --name url python=3.10
pip install -r requirements.txt
```

Files must be run from the root directory of the repository. 

## Downloading and formatting datasets

```bash
# Download the gas sensor dataset
python get_real_dataset/get_data.py -d https://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip -n gas_sensor

# Format the gas sensor dataset to csv
python get_real_dataset/format_data.py
```

## Recursive partition-based K-means implementation

The RPKM implementation is located in `./src/rpbkmeans.py`.

## Load and create datasets

`dataset.py` and `artificial_dataset_generator.py` files are used to load and create datasets.

## Experiments

Experiment scripts are located in the `./src/experiments/` folder. They generate csv result files which are stored in the `./results/` folder by default. The code to generate the plots and images from the csv files are inside the `./src/experiments/plots/` folder.

## Report

The report is located in `./doc/` folder.