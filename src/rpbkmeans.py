import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from src.subset import Subset

class RecursivePartitionBasedKMeans(BaseEstimator, ClusterMixin, ClassifierMixin):
    """ Recursive Partition Based K-Means clustering algorithm.
    
    Parameters
    ----------
    k : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    
    max_iterations : int, default=6
        The maximum number of iterations to perform.
    
    distance_threshold : float, default=0
        The threshold for the distance between the previous centroids and the new centroids.
    
    Attributes
    ----------
    centroids : np.ndarray of shape (k, n_features)
        The cluster centroids.
        
    distance_computations_ : int
        The number of distance computations performed.
        
    instance_ratio_ : float
        The ratio of instances to the total number of samples.
        
    labels_ : np.ndarray of shape (n_samples,)
        The cluster labels for each sample.
    
    n_iterations : int
        The number of iterations performed.
        
    """
    def __init__(self, k=8, max_iterations=6, distance_threshold=0):
        self.reset()
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = distance_threshold
        self.instance_ratio_ = -1
        self.distance_computations_ = 0
        self.labels_ = None

    def reset(self):
        """ Reset the attributes of the estimator. """
        self.centroids = None
        self.distance_computations_ = 0
        self.instance_ratio_ = -1
        self.labels_ = None
        self.n_iterations = 0
    
    def _create_recursive_partition(self, data_points, current_partitions):
        """ Create a recursive partition based on the data points and current partitions.
        
        Parameters
        ----------
        data_points : np.ndarray
            The data points to partition.
        current_partitions : list
            The current partitions.

        Returns
        -------
        new_partitions : list
            The new partitions based on the data points and current partitions.
        """
        new_partitions = []
        for partition in current_partitions:
            new_partitions.extend(self._perform_partition(data_points, partition))
        return new_partitions

    def _perform_partition(self, data_points, partition):
        """ Partition the data based on threshold comparisons. 
        
        Parameters
        ----------
        data_points : np.ndarray
            The data points to partition.
        partition : Subset
            The current partition.
        
        Returns
        -------
        new_partitions : list
            The list of new partitions based on the threshold comparisons.
        """
        partitioned_data = data_points[partition.indexes]
        threshold_comparisons = partitioned_data > partition.thresholds[None, :]
        hypercube_indices = (threshold_comparisons * (2 ** np.arange(threshold_comparisons.shape[1]))).sum(axis=1)
        return self._generate_sub_partitions(hypercube_indices, partition)

    def _generate_sub_partitions(self, hypercube_indices, partition):
        """ 
        Generate sub-partitions based on the hypercube index from binary comparisons. 
        
        Parameters
        ----------
        hypercube_indices : np.ndarray
            The hypercube indices for the partition.
        partition : Subset
            The current partition.
        
        Returns
        -------
        new_partitions : list
            The list of new partitions based on the hypercube indices."""
        new_partitions = []
        unique_indices = sorted(set(hypercube_indices))
        for index in unique_indices:
            partition_indices = partition.indexes[hypercube_indices == index]
            updated_max = partition.max.copy()
            updated_min = partition.min.copy()
            updated_thresholds = partition.thresholds.copy()
            binary_representation = (index & (2 ** np.arange(len(partition.max)))) > 0
            updated_min, updated_max = self._update_ranges(binary_representation, updated_min, updated_max, updated_thresholds)
            new_mid_thresholds = (updated_max + updated_min) / 2
            new_partitions.append(Subset(partition_indices, updated_max, updated_min, new_mid_thresholds))
        return new_partitions

    def _update_ranges(self, binary_representation, min_values, max_values, thresholds):
        """ 
        Adjust min and max ranges based on the binary representation of the partition index. 
        
        Parameters
        ----------
        binary_representation : np.ndarray
            Binary representation of the partition index.
        min_values : np.ndarray
            Minimum values of the partition.
        max_values : np.ndarray
            Maximum values of the partition.
        thresholds : np.ndarray
            Threshold values for partitioning in the RPBM algorithm.
            
        Returns
        -------
        min_values : np.ndarray
            Updated minimum values of the partition.
        max_values : np.ndarray
            Updated maximum values of the partition.
        """
        min_values[binary_representation] = thresholds[binary_representation]
        max_values[~binary_representation] = thresholds[~binary_representation]
        return min_values, max_values
    
    def fit(self, X, y=None):
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X, y=y, ensure_2d=True) # Validate the input data
        n_samples = X.shape[0]

        self.reset() # Reset the attributes of the estimator

        partition = [Subset(np.arange(n_samples), np.full((X.shape[1],), 1), np.full((X.shape[1],), -1), np.full((X.shape[1],), 0))]
        iteration = 0
        while iteration < self.max_iterations:
            partition = self._create_recursive_partition(X, partition)

            representatives, weights = np.empty((len(partition), self.n_features_in_)), np.empty((len(partition),))

            for i, p in enumerate(partition): # Compute the representative and weights for each partition
                representatives[i] = p.representative(X)
                weights[i] = len(p)

            if len(partition) < self.k: 
                continue

            elif self.centroids is None:
                centers = np.random.choice(range(representatives.shape[0]), self.k, replace=False) # Randomly select the centroids
                self.centroids = representatives[centers]

            elif len(partition) >= n_samples:
                break

            prev_centroids = self.centroids.copy() 
            km = KMeans(
                n_clusters=self.k, 
                init=self.centroids, # Use the previous centroids as the initialization
                n_init=1,
                algorithm='lloyd', # Lloyd's algorithm
            ).fit(
                representatives,
                sample_weight=weights
            ) # Fit the KMeans model
            self.centroids = km.cluster_centers_
            self.distance_computations_ += km.n_iter_ * self.k * representatives.shape[0]
            iteration += 1
            
            if np.allclose(self.centroids, prev_centroids, atol=self.tolerance): # Check if the centroids have converged
                break

        self.instance_ratio_ = len(partition) / n_samples
        #self.labels_ = self.predict(X)
        self.n_iterations = iteration
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        
        Returns
        -------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        if self.centroids is None:
            raise ValueError("Model not fitted yet")

        X = np.array(X)

        if self.centroids.shape[1] != X.shape[1]:
            raise ValueError(f"Expected {self.centroids.shape[1]} features, got {X.shape[1]}")

        return pairwise_distances_argmin(X, self.centroids)

# import numpy as np
# from sklearn.datasets import make_blobs

# # Generate synthetic data
# X, _ = make_blobs(n_samples=1300, centers=3, n_features=2)

# # Instantiate and fit the RPKM model
# model = RecursivePartitionBasedKMeans(k=3, max_iterations=6)
# a = model.fit(X)

# # Predict cluster labels for the same dataset
# labels = model.predict(X)

# # Plotting the results
# import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
# plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=100)
# plt.title('Recursive Partition Based K-Means Clustering')
# plt.show()