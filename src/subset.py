import numpy as np

class Subset:
    def __init__(self, indexes) -> None:
        self.indexes = indexes

    def __len__(self) -> int:
        return len(self.indexes)

    def get_representative(self, data):
        return np.mean(data[self.indexes], axis=0)

    


    