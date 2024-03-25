import numpy as np

from subset import Subset

class RecursivePartitionBasedKMeans:
    def __init__(self, k=4, random_state=0) -> None:
        self.max_iterations = 100
        self.k = k
        self.centroids = None
        self.random_state = random_state

    def _create_partition(self, X, partition):
        """Create a quadtree partition of the data."""
        new_partition = []

        for subset in partition:

            # If the subset is empty, skip it
            if len(subset) == 0:
                continue

            # Calculate the centroid of the subset
            centroid = np.mean(X[subset.indices], axis=0)

            # Calculate the distance of each point to the centroid
            distances = np.linalg.norm(X[subset.indices] - centroid, axis=1)

            # Find the point with the maximum distance
            max_distance_index = np.argmax(distances)

            # Split the subset into two subsets
            left_subset = Subset(subset.indices[:max_distance_index], subset.start, subset.start + max_distance_index)
            right_subset = Subset(subset.indices[max_distance_index:], subset.start + max_distance_index, subset.end)

            new_partition.append(left_subset)
            new_partition.append(right_subset)

        return new_partition


    def fit(self, X, threshold=0.01):
        X = X.copy()

        i = 0
        partition = [Subset(X, 0, X.shape[0])]

        while i < self.max_iterations:
            
            partition = self._create_partition(X, partition)

            # Get representatives and weights for computed partition
            representatives, W = [(subset.get_representative(X[subset.indexes]), len(subset)) for subset in partition]
            
            # Initialize centroids
            np.array(representatives[np.random.choice(representatives.shape[0], self.k, replace=False)])


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