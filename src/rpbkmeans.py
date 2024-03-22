import numpy as np

from subset import Subset

class RecursivePartitionBasedKMeans:
    def __init__(self) -> None:
        self.centroids = None
        self.max_iterations = 100

    def _create_partition(self, X, k):
        pass


    def fit(self, X, k, threshold=0.01):
        X = X.copy()

        i = 0

        while i < self.max_iterations:
            
            self._create_partition(X, k)

            i += 1

        return 

    def predict(self, X):
        pass

    """def weighted_lloyd(self, X, w, k, centroids, threshold=0.01, max_iterations=100):
        i = 0
        while i < max_iterations:
            # Assign each point to the nearest centroid
            clusters = self.assign_points_to_clusters(X, centroids)

            # Calculate the new centroids
            new_centroids = self.calculate_new_centroids(X, w, clusters, k)

            # Check for convergence
            if np.max(self.centroids - np.array(new_centroids)):
                break 

            centroids = new_centroids
            i += 1"""

if __name__ == "__main__":
    X = np.random.rand(10, 2)
    rpbkmeans = RecursivePartitionBasedKMeans()
    rpbkmeans.fit(X, 3)
    print(rpbkmeans.predict(X))