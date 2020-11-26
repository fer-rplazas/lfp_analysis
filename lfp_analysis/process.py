
import numpy as np
import pandas as pd
import h5py
import mne
from librosa.util import frame
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from pathlib import Path


class TfTransformer:
    """Time-frequency transformer
    """

    def __init__(self, data):
        self.frequencies = np.arange(2, 101, 1)
        self.n_cycles = self.frequencies / 3
        self.LFPs = data

    @classmethod
    def from_importer(cls, signal_bundle): return cls(signal_bundle.data)

    def TF(self):
        self.power = mne.time_frequency.tfr_array_morlet(
            self.LFPs[np.newaxis, ...],
            n_cycles=self.n_cycles,
            freqs=self.frequencies,
            decim=1,
            output="power",
            sfreq=2048,
        ).squeeze()

        return self

    def normalize(self):
        mean = np.nanmean(self.power, axis=-1, keepdims=True)
        self.power = (self.power - mean) / mean

        return self

    def zscore(self):
        mean = np.nanmean(self.power, axis=-1, keepdims=True)
        std = np.nanstd(self.power, axis=-1, keepdims=True)
        self.power = (self.power - mean) / std

        return self

    def range_correct(self):
        self.range_corrected = (
            self.limited / self.limited.max(-1, keepdims=True) * 2 - 1
        )

        return self


class AxisApplier:
    """Applies function f along first dimension.

    Intended for structures of shape (channels, frequencies, times) where f = f(frequencies,times) (appy to each channel)
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, X, **kwargs):
        for i in range(X.shape[0]):
            X[i] = self.f(X[i], **kwargs)

        return X


def zscore(X):
    return (X - np.mean(X, axis=-1, keepdims=True)) / np.std(X, axis=-1, keepdims=True)


def bl_correct(X):
    mean = np.mean(X, axis=-1, keepdims=True)
    return (X - mean) / mean


def clip(X, std_factor=7):
    return Normalize(vmin=-std_factor, vmax=std_factor, clip=True)(X)
