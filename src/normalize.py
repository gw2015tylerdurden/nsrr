from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod
import numpy as np

class Normalizer(ABC):

    @abstractmethod
    def normalize(self, data):
        pass

    @classmethod
    def get_instance(cls, method):
        if method == 'standard':
            return StandardScalerNormalizer()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

class StandardScalerNormalizer(Normalizer):
    def __init__(self):
        self.ss = None

    def normalize(self, data):
        data = np.array(data).reshape(-1, 1)
        if self.ss is None:
            ss = StandardScaler()
        return ss.fit_transform(data).ravel()

