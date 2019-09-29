import sys
import numpy as np
import pandas as pd
import scipy
import librosa
import sklearn
from sklearn.utils import shuffle
import random

def wavtospec(mixture, hop_size, window_size, window):

    STFT_mixture = librosa.core.stft(mixture, hop_length=hop_size, n_fft=window_size, window=window)
    data = 2 * np.abs(STFT_mixture) / np.sum(window)
    data = np.reshape(data, (257, 626, 1))

    return data

def checkNoise(mixture):

    win_len=4096
    lpf_cutoff=0.075
    theta=0.15
    var_lambda=20.0
    amplitude_threshold=0.01
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]

    mixture = track_energy(mixture, win_len, win_len // 2, win)
    E0 = np.sum(mixture, axis=0)
    mixture[E0 < amplitude_threshold] = 0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, 'low')
    mixture = scipy.signal.filtfilt(b, a, mixture)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (mixture - theta)))))

    return np.mean(C)

def spectowav(inp, hop_size, window, phase):

    inp_spec = np.abs(inp)
    inp_spec = np.multiply(inp_spec, phase)
    inp_istft = librosa.core.istft(inp_spec, hop_length=hop_size, window=window)
    #inp_istft = scipy.signal.wiener(inp_istft)

    return inp_istft

def track_energy(wave, win_len, hop_len, win):
    """Compute the energy of an audio signal
    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation
    Returns
    -------
    energy : np.array
        Array of track energy
    """

    wave = np.lib.pad(
        wave, pad_width=(win_len-hop_len, 0), mode='constant', constant_values=0
    )

    # post padding
    wave = librosa.util.fix_length(
        wave, int(win_len * np.ceil(len(wave) / win_len))
    )

    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression

    return np.mean((wavmat.T * win), axis=1)


def hwr(x):
    """ Half-wave rectification.
    Parameters
    ----------
    x : array-like
        Array to half-wave rectify
    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array
    """
    return (x + np.abs(x)) / 2
