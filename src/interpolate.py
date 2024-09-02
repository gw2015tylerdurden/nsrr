from scipy.interpolate import interp1d, Akima1DInterpolator
from abc import ABC, abstractmethod
import numpy as np


class Interpolator(ABC):

    @abstractmethod
    def interpolate(self, y, x, x_new):
        pass

    @classmethod
    def get_instance(cls, kind):
        if kind in ['cubic', 'linear', 'akima']:
            return Interpolator1d(kind)
        else:
            raise ValueError(f"Unknown interpolation method: {kind}")


class Interpolator1d(Interpolator):
    def __init__(self, kind):
        self.kind = kind

    def interpolate(self, y, x, x_new):
        if self.kind == 'akima':
            f = Akima1DInterpolator(x, y)
        else:
            f = interp1d(x, y, kind=self.kind)
        return f(x_new)
