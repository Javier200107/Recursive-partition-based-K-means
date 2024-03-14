import numpy as np

class RecursivePartitionBasedKMeans:
    def __init__(self) -> None:
        pass

    def fit(self, X, k):
        pass

    def predict(self, X):
        pass

if __name__ == "__main__":
    X = np.random.rand(100, 2)
    rpbkmeans = RecursivePartitionBasedKMeans()
    rpbkmeans.fit(X, 3)
    print(rpbkmeans.predict(X))