import logging
import random
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from artificial_dataset_generator import ArtificialDatasetGenerator
from src.rpbkmeans import RecursivePartitionBasedKMeans as RPKM
from sklearn.cluster import KMeans, MiniBatchKMeans

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Constants
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# Output folder configuration
OUT_FOLDER = Path('results/')
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Experiment configurations
N_REPLICAS = 10
N_CLUSTERS = [3, 9]
N_DIMS = [2, 4, 8]
N_SAMPLES = [100, 1000, 10000, 100000, 1000000]
N_THREADS = None  # Use 1 or more threads, if None, all CPUs are used.

# Algorithm parameters for testing
KWARGS = [
    {"algorithm": 'k-means++', "param": 0},
    {"algorithm": 'rpkm', "param": 1},
    {"algorithm": 'rpkm', "param": 2},
    {"algorithm": 'rpkm', "param": 3},
    {"algorithm": 'rpkm', "param": 4},
    {"algorithm": 'rpkm', "param": 5},
    {"algorithm": 'rpkm', "param": 6},
    {"algorithm": "mb-kmeans", "param": 100},
    {"algorithm": "mb-kmeans", "param": 500},
    {"algorithm": "mb-kmeans", "param": 1000},
]

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def set_logger(debug=False):
    """Set logger configuration."""
    log_file_path = OUT_FOLDER / 'artificial_dataset_tests.log'
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(module)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    logging.basicConfig(filename=log_file_path, level=level,
                        format=logging_format)
    if debug:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(logging_format))
        logging.getLogger().addHandler(consoleHandler)

def compute(n_clusters, n_dims, n_samples, k_means_kwargs, computationidx):
    """Compute the distance calculations for a given configuration."""
    logging.info(f'Computing: n_clusters={n_clusters}, n_dims={n_dims}, n_samples={n_samples}, kwargs={k_means_kwargs}, idx={computationidx}')
    np.random.seed(computationidx)
    random.seed(computationidx)

    dataset_generator = ArtificialDatasetGenerator(
        n_centers=n_clusters,
        n_features=n_dims,
        n_samples=int(n_samples),
        normalize=True,
        n_replicas=1)

    data, _ = dataset_generator()
    start_time = time.time()

    if k_means_kwargs["algorithm"] == 'rpkm':
        clf = RPKM(k=n_clusters, max_iterations=k_means_kwargs['param']).fit(data)
        distance_calculations = clf.distance_computations
    elif k_means_kwargs["algorithm"] == 'mb-kmeans':
        clf = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1, batch_size=k_means_kwargs['param']).fit(data)
        distance_calculations = n_clusters * clf.n_iter_ * data.shape[0]
    else:
        clf = KMeans(n_clusters=n_clusters, init=k_means_kwargs["algorithm"], n_init=1).fit(data)
        distance_calculations = n_clusters * data.shape[0] * clf.n_iter_

    fit_time = time.time() - start_time
    logging.info(f"Completed: Distance calculations={distance_calculations}, Time={fit_time}")
    return n_clusters, n_dims, n_samples, k_means_kwargs['algorithm'], k_means_kwargs['param'], distance_calculations, fit_time

def main():
    """Main function to run the experiments."""
    set_logger(debug=False)
    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES, KWARGS, range(N_REPLICAS)))
    print(f"Running {len(combinations)} combinations")

    if N_THREADS == 1:
        results = []
        for args in tqdm(combinations):
            results.append(compute(*args))
    else:
        print(f"Running with {N_THREADS} threads")
        results = []
        pb = tqdm(total=len(combinations))

        def update(ans):
            results.append(ans)
            pb.update()

        with Pool(N_THREADS) as p:
            for comb in combinations:
                p.apply_async(compute, args=comb, callback=update)
            p.close()
            p.join()
        pb.close()
    results.sort()

    columns = ['n_clusters','n_dims','n_samples'] + list(KWARGS[0].keys()) + ['distance_calculations_average','fit_time_average']
    results = pd.DataFrame(results, columns=columns)
    results = results.fillna('None')
    results = results.groupby(columns[:5]).mean().reset_index()
    results.to_csv(OUT_FOLDER / 'distances_artificial_agg.csv', sep=',', index=False)

if __name__ == '__main__':
    main()
