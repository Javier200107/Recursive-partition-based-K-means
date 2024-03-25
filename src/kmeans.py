import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KMeans:

    """Class to perform K-Means Clustering"""

    def __init__(self, k=3, random_state=0):
        """Initialize the KMeansClustering class"""
        self.k = k
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0
        self.random_state = random_state
        
    @staticmethod
    def euclidean_distance(data_point, centroids):
        """Calculate the Euclidean distance between a data point and the centroids"""
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))
    
    def fit(self, data, max_iterations=100, use_points_as_centroids=True):
        self.fit_predict(data, max_iterations, use_points_as_centroids)

    def fit_predict(self, data, max_iterations=100, use_points_as_centroids=True, tolerance = 0.0001):
        """Train the K-Means Clustering model"""
        np.random.seed(self.random_state)
        # Convert the input data to a numpy array
        X = data.copy()
        X = data.to_numpy()
        
        if use_points_as_centroids:
            # Initialize random data points to centroids
            self.centroids = np.array(X[np.random.choice(X.shape[0], self.k, replace=False)])
        else:
            # Create random centroids
            self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for _ in range(max_iterations):
            idxs = []
            self.inertia_ = 0
            # Assign each data point to the closest centroid
            for data_point in X:
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster_number = np.argmin(distances)
                idxs.append(cluster_number)
                self.inertia_ += distances[cluster_number]

            idxs = np.array(idxs)
            cluster_idxs = []

            for i in range(self.k):
                cluster_idxs.append(np.argwhere(idxs == i))
            new_centroids = []

            # Calculate the new centroids
            for i, indexes in enumerate(cluster_idxs):
                if len(indexes) == 0:
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(np.mean(X[indexes], axis=0)[0])
            
            # Check if the centroids have converged
            if np.max(self.centroids - np.array(new_centroids)) < tolerance:
                break
            else:
                self.centroids = np.array(new_centroids)
        # Convert idxs to a DataFrame column
        data["label"] = idxs
        self.labels_ = idxs
        return data
    
    def evaluate(self, data, labels):
        """Evaluate the K-Means Clustering model"""
        pass


if __name__ == "__main__":
    data = np.random.rand(10, 2)
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    print(kmeans.centroids)

    # Plot the data points and centroids
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='r')

    plt.show()