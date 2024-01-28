from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import numpy as np


class Interpolator(ABC):

    @abstractmethod
    def interpolate(self, y, x, x_new, kind):
        pass

    @classmethod
    def get_instance(cls, method):
        if method in ['cubic', 'linear']:
            return Interpolator1d()
        else:
            raise ValueError(f"Unknown interpolation method: {method}")


class Interpolator1d(Interpolator):

    def interpolate(self, y, x, x_new, kind='linear'):
        f = interp1d(x, y, kind=kind)
        return f(x_new)
