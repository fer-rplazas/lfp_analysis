
# __all__ = ['SmrImporter', 'SignalBundle', 'AccData',
#            'LFPData', 'TfTransformer', 'AxisApplier',
#            'zscore', 'bl_correct', 'clip']

from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from neo.io import Spike2IO
from scipy.signal import butter, filtfilt, decimate, iirnotch
from scipy.ndimage.filters import uniform_filter1d

from textwrap import dedent

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


###############################################################################
# Main Classses:
###############################################################################


class SmrImporter:
    """Reads signals and metadata (sampling frequency ``fs``, channel names ``ch_names``)from SMR file

    """

    def __init__(self, fname):
        analog_signal = (
            Spike2IO(filename=str(fname)).read()[
                0].segments[0].analogsignals[0]
        )

        self.ch_names = np.vectorize(lambda x: x.decode("utf-8"))(
            analog_signal.channel_index.channel_names
        ).tolist()

        self.ch_dict = {name: i for i, name in enumerate(self.ch_names)}

        self.fs = analog_signal.sampling_rate.magnitude

        self.data = analog_signal.magnitude.T

    def free(self):
        if hasattr(self, "data"):
            del self.data


class SignalBundle:
    """Superclass to represent bundles of signals that are often grouped together (e.g. bipolar LFPs, etc.)

    """

    def __init__(self, data, names, fs=2048):
        self.names = names
        self.data = data if data.shape[1] > data.shape[0] else data.T
        self.fs = float(fs)

        self.highpassed, self.decimated = False, False

    def __repr__(self):

        reprStr = f"""
        SignalBundle with
        
        - Channel Names: {self.names}
        - Shape: {self.data.shape}
        - End Time: {self.data.shape[1]/float(self.fs):.3f} s
        
        """
        return dedent(reprStr)

    def highpass(self, cutoff_freq=1):
        b, a = butter(4, cutoff_freq, "hp", fs=self.fs)
        self.data = filtfilt(b, a, self.data, axis=-1)

        self.highpassed = True

        return self

    def channel_apply(self, f, **kwargs):
        if len(self.data.shape) == 1:
            self.data = f(self.data, **kwargs)
            return self

        self.data = np.apply_along_axis(f, -1, self.data, **kwargs)
        return self

    def limit(self, plot=False, **kwargs):

        if plot:  # Plot before limiting
            n_chan = len(self.names)
            time = self.time()

            fig, ax = plt.subplots(n_chan, 1, sharex=True, figsize=(13, 6))
            ax = ax.flatten() if n_chan > 1 else [ax]

            for i in range(n_chan):
                if n_chan > 1:
                    ax[i].plot(time, self.data[i], alpha=0.5, label="Before")
                else:
                    ax[i].plot(time, self.data, alpha=0.5, label="Before")

        self.channel_apply(limiter, **kwargs)  # Limit signals

        if plot:  # Plot after limiting
            for i in range(n_chan):
                if n_chan > 1:
                    ax[i].plot(time, self.data[i], label="After")
                else:
                    ax[i].plot(time, self.data, label="After")

                ax[i].set_title(self.names[i])
                ax[i].legend()
            plt.show()

        return self

    def decimate(self, factor=2):
        self.data = decimate(self.acc, factor)
        self.decimated = True

        return self

    def time(self):
        if len(self.data.shape) > 1:
            return np.linspace(0, self.data.shape[1] / self.fs, self.data.shape[1])
        else:
            return np.linspace(0, self.data.shape[0] / self.fs, self.data.shape[0])


class AccData(SignalBundle):
    """Accelerometer SignalBundle
    """
    @classmethod
    def from_importer(
        cls, importer, acc_names=[
            "Aclx", "Acly", "Aclz", "Acrx", "Acry", "Acrz"]
    ):
        data = importer.data[[importer.ch_dict[name] for name in acc_names], :]
        return cls(data, acc_names, importer.fs)

    def to_magnitude(self):
        if self.data.shape[1] == 6:
            self.data = np.linalg.norm(self.data[:3, ], axis=0) + np.linalg.norm(
                self.data[3:, ], axis=0
            )
        else:
            self.data = np.linalg.norm(self.data, axis=0)

        self.names = ["Acc_merged"]

        return self

    def to_label(self, percentile=80, plot=True):

        if not self.highpassed:
            _ = self.highpass()

        if len(self.data.shape) > 1:
            raise RuntimeError(
                "Accelerometer labels have to be computed from single-channel signal. Please call .to_magnitude() method to get single accelerometer channel."
            )

        smoothed = uniform_filter1d(self.data, size=int(self.fs // 32))

        thresh = np.percentile(smoothed, percentile)
        self.label = (np.abs(smoothed) > thresh).astype(np.float)

        label_smoothed = uniform_filter1d(self.label, size=int(self.fs))
        # assert np.allclose(self.label, label_smoothed)

        self.label = (label_smoothed > 0.5).astype(np.float)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(13, 5))

            ax1.plot(self.time(), smoothed)
            ax1.plot(self.time(), self.label * thresh * 5)

            ax2.plot(self.time(), self.data)

        return self


class LFPData(SignalBundle):
    """LFP data
    """

    @classmethod
    def from_importer(cls, importer, names):
        data = importer.data[[importer.ch_dict[name] for name in names], :]
        return cls(data, names, importer.fs)


###############################################################################
# Utilities and other functions:
###############################################################################


def limiter(x, method="std", percentiles=(0.1, 99.99), std_factor=5, values=None):
    """Limits 1-d signal x using one of the following methods: 

        - "std" limits signal with values: (-std_factor * x.std(), std_factor * x.std())
        - "values" limits signal with values specified by user in tuple values
        - "percentile" limits signal according to percentiles specified in tuple percentiles        

    """

    if method == "values":
        if values == None:
            raise ValueError(
                "If you choose 'values' as method, you have to specify the 'values' parameter as a tuple: (lower_bound, higher_bound)"
            )
        else:
            return np.clip(x, values[0], values[1])

    if method == "std":
        std = std_factor * x.std()
        return np.clip(x, -std, std)

    elif method == "percentile":
        ranges = np.percentile(x, percentiles)
        return np.clip(x, -ranges[0], ranges[1])

    else:
        raise ValueError(
            f"`method` has to be one of: ('std', 'percentile') –> you passed {method}"
        )


def limiter2d(x):
    return np.apply_along_axis(limiter, -1, x)


def save_to_h5(path, lfp, label):
    f = h5py.File(path, "w")
    f.create_dataset("LFP", data=lfp)
    f.create_dataset("label", data=label)
    f.close()
