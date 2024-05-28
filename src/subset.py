import numpy as np
from typing import List, Optional

class Subset:
    """
    Class to store the indexes of the data points in the subset along with the max, min, 
    and thresholds of the subset. Provides functionality to compute the representative mean
    of the subset.
    """
    def __init__(self, indexes: List[int], max_: np.ndarray, min_: np.ndarray, thresholds: np.ndarray) -> None:
        """
        Initializes the Subset with specified indexes, maximum, minimum, and thresholds,
        converting these properties to float32 for consistent numerical precision and lower memory usage.

        Parameters:
        - indexes (List[int]): Indexes of the subset elements.
        - max_ (np.ndarray): The maximum boundary array for the subset.
        - min_ (np.ndarray): The minimum boundary array for the subset.
        - thresholds (np.ndarray): Threshold values for partitioning in the RPBM algorithm.
        """
        self.indexes = np.array(indexes, dtype=int)
        self.max = max_.astype(np.float32)
        self.min = min_.astype(np.float32)
        self.thresholds = thresholds.astype(np.float32)

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the subset showing min, max, and thresholds.

        Returns:
        - str: Human-readable representation of the Subset object.
        """
        return f"{self.__class__.__name__}: [min={self.min}, max={self.max}, thresholds={self.thresholds}]"

    def __repr__(self) -> str:
        """ Returns the official string representation of the Subset object. """
        return self.__str__()

    def __len__(self) -> int:
        """
        Returns the number of indexes in the subset.

        Returns:
        - int: The number of data points in the subset.
        """
        return len(self.indexes)

    def representative(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculates the representative (mean) of the subset data points if the subset is non-empty.

        Parameters:
        - data (np.ndarray): The dataset from which to gather data points using the indexes of this subset.

        Returns:
        - np.ndarray: Mean of the data points in the subset if non-empty, otherwise None.
        """
        if self.indexes.size > 0:
            return np.mean(data[self.indexes], axis=0)
        return None
