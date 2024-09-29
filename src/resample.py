from scipy.interpolate import interp1d, Akima1DInterpolator
from abc import ABC, abstractmethod
import scipy.signal as signal


class Resample(ABC):

    @abstractmethod
    def upsampling(self, y, x, x_new, fs=None, target_fs=None):
        pass

    @classmethod
    def get_instance(cls, kind):
        if kind in ['cubic', 'linear', 'akima', 'fft']:
            return Resample1d(kind)
        else:
            raise ValueError(f"Unknown interpolation method: {kind}")


class Resample1d(Resample):
    def __init__(self, kind):
        self.kind = kind

    def upsampling(self, y, x, x_new, fs=None, target_fs=None):
        if self.kind == 'fft':
            if fs is None:
                raise ValueError("Incorrect sampling frequency for resampling")
            #return signal.resample(y, len(x_new), window=signal.get_window('hamming', len(y)))
            up, down = None, None
            if int(fs) == 1:
                up, down = int(target_fs), int(fs)
            elif int(fs) == 10:
                up, down = 25, 2
            elif int(fs) == 50:
                up, down = 5, 2
            elif target_fs == fs:
                # 125 Hz
                return y
            #return signal.resample_poly(y, up, down)
            return signal.resample_poly(y, up, down, window=signal.get_window('hamming', len(x_new)))


        if self.kind == 'akima':
            f = Akima1DInterpolator(x, y)
        else:
            f = interp1d(x, y, kind=self.kind)
        # 8 order and cutoff is Nyquist
        if target_fs == fs:
            # no need to apply low pass filter
            return y
        else:
            if fs == 1.0:
                order = 1
            else:
                order = 4
            b, a = signal.butter(order, (fs*0.5) / (target_fs*0.5))
            return signal.filtfilt(b, a, f(x_new))
