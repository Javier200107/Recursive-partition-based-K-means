import numpy as np
import numpy.ma as ma
from sklearn import datasets
from statistics import NormalDist
from scipy.spatial.distance import cdist


class ArtificialDatasetGenerator:
    def __init__(self, n_centers: int, n_features: int, n_samples: int,
                 normalize: bool = True, n_replicas: int = -1, **dataset_kwargs):
        """
        Initialize the dataset generator.
        
        Parameters:
        - n_centers: Number of cluster centers.
        - n_features: Number of features per sample.
        - n_samples: Total number of samples.
        - normalize: Flag to normalize the dataset.
        - n_replicas: Number of times the dataset can be regenerated.
        - dataset_kwargs: Additional keyword arguments for make_blobs.
        """
        self.n_centers = n_centers
        self.n_features = n_features
        self.n_samples = n_samples
        self.normalize = normalize
        self.n_replicas = n_replicas
        self.dataset_kwargs = dataset_kwargs
        self.max_overlap = 0.05  # Maximum allowed overlap between clusters.

    def __call__(self):
        """Generate the dataset and normalize it if required."""
        x, y = self.generate_dataset()
        if self.normalize:
            x = self.normalize_dataset(x)
        return x, y

    def __getitem__(self, idx):
        """Enable indexing to regenerate datasets, limited by n_replicas."""
        if 0 <= self.n_replicas <= idx:
            raise IndexError("Replica index out of range")
        return self()

    def __str__(self):
        """String representation of the class configuration."""
        return (f"ArtificialDatasetGenerator(n_centers={self.n_centers}, n_features={self.n_features}, "
                f"n_samples={self.n_samples}, normalize={self.normalize}, n_replicas={self.n_replicas})")

    @property
    def shape(self):
        """Return the shape of the dataset as a tuple."""
        return (self.n_samples, self.n_features), (self.n_samples,)

    def generate_dataset(self):
        """Generate a synthetic dataset using the make_blobs function from sklearn."""
        # Randomly generate cluster centers within the range [-1, 1] for each feature.
        centers = 2 * (np.random.rand(self.n_centers, self.n_features) - 0.5)
        sigma = self.find_std(centers)  # Determine the standard deviation for the clusters.
        return datasets.make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=centers,
            cluster_std=sigma,
            **self.dataset_kwargs)

    def normalize_dataset(self, x):
        """Normalize the dataset to have features in the range [0, 1]."""
        return x / np.max(np.abs(x), axis=0, keepdims=True)

    def find_std(self, centers):
        """Calculate a suitable standard deviation for clusters to control overlap."""
        # Compute pairwise distances between cluster centers.
        center_distances = cdist(centers, centers)
        min_dist = ma.masked_equal(center_distances, 0).min()  # Ignore zero distance (self-distance).
        sigma = min_dist / 3  # Start with a third of the minimum distance between centers.

        try:
            # Adjust sigma to achieve the desired maximum overlap between clusters.
            target_overlap = self.max_overlap
            for _ in range(10):  # Cap the number of adjustments to prevent infinite loops.
                dist1 = NormalDist(mu=0, sigma=sigma)
                dist2 = NormalDist(mu=min_dist, sigma=sigma)
                overlap = dist1.overlap(dist2)

                if np.isclose(overlap, target_overlap, atol=0.01):
                    break  # Acceptable overlap reached.

                if overlap > target_overlap:
                    sigma /= 1 + (overlap - target_overlap)
                else:
                    sigma *= 1 + (target_overlap - overlap)

        except OverflowError:
            print("Overflow in calculating sigma, using last valid sigma")
        return sigma

if __name__ == '__main__':
    generator = ArtificialDatasetGenerator(4, 2, 100)
    x, y = generator()
    print('X:', x)
    print('Y:', y)
