# JACSNet
# Author: Vanessa H. Tan 09.21.19

# get libraries
import sys
import numpy as np
import pandas as pd
import scipy
import librosa
import sklearn
from sklearn.utils import shuffle
import random

import keras
from keras.utils import Sequence
from utils import wavtospec, checkNoise, _augment_gain

class Generator(Sequence):
    def __init__(self, batch_size, shuffle, mode, dim=(1025, 94), n_channels=1, n_classes=4):
        self.mode = mode
        if self.mode is "train":
            self.data_dir = "D:/Data/Musdb18/16000_3s/train/"
            self.csv_file = pd.read_csv("D:/Data/Musdb18/16000_3s/train.csv")
        elif self.mode is "test":
            self.data_dir = "D:/Data/Musdb18/16000_3s/test/"
            self.csv_file = pd.read_csv("D:/Data/Musdb18/16000_3s/test.csv")
        elif self.mode is "valid":
            self.data_dir = "D:/Data/Musdb18/16000_3s/valid/"
            self.csv_file = pd.read_csv("D:/Data/Musdb18/16000_3s/valid.csv")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.csv_file.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        filename = [self.csv_file["fname"][k] for k in indexes]
        number = [self.csv_file["num"][k] for k in indexes]

        # Generate data
        data, label = self.__data_generation(filename, number)
        return data, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.csv_file.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, name, num):

        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        X_input = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, self.n_classes))

        sourceclass = np.zeros((self.batch_size, self.n_classes), dtype=int)

        window_size = 2048
        hop_size = 512
        window = np.blackman(window_size)

        for i, (filename, number) in enumerate(zip(name, num)):

            STFTclass = [0, 0, 0, 0]
            # mixture, sr = librosa.load(str(self.data_dir + filename + '/' + filename + str(number) + '.wav'), mono=True, sr=16000)
            # data = wavtospec(mixture, hop_size, window_size, window)
            # X[i,] = data.astype('float32')

            vocals, sr = librosa.load(str(self.data_dir + filename + '/' + filename + str(number) + '_vocals.wav'), mono=True, sr=16000)
            if self.mode == "train":
                vocals = _augment_gain(vocals)
            STFT_vocals = wavtospec(vocals, hop_size, window_size, window)
            if ((checkNoise(vocals) > 0.05)): STFTclass[0] = 1

            bass, sr = librosa.load(str(self.data_dir + filename + '/' + filename + str(number) + '_bass.wav'), mono=True, sr=16000)
            if self.mode == "train":
                bass = _augment_gain(bass)
            STFT_bass = wavtospec(bass, hop_size, window_size, window)
            if ((checkNoise(bass) > 0.05)): STFTclass[1] = 1

            drums, sr = librosa.load(str(self.data_dir + filename + '/' + filename + str(number) + '_drums.wav'), mono=True, sr=16000)
            if self.mode == "train":
                drums = _augment_gain(drums)
            STFT_drums = wavtospec(drums, hop_size, window_size, window)
            if ((checkNoise(drums) > 0.05)): STFTclass[2] = 1

            others, sr = librosa.load(str(self.data_dir + filename + '/' + filename + str(number) + '_others.wav'), mono=True, sr=16000)
            if self.mode == "train":
                others = _augment_gain(others)
            STFT_others = wavtospec(others, hop_size, window_size, window)
            if ((checkNoise(others) > 0.05)): STFTclass[3] = 1

            mix = vocals + bass + drums + others
            inputdata = wavtospec(mix, hop_size, window_size, window)
            X_input[i,] = inputdata.astype('float32')

            STFT = [STFT_vocals, STFT_bass, STFT_drums, STFT_others]

            label = np.concatenate([STFT[0], STFT[1], STFT[2], STFT[3]], axis=2)
            y[i,] = label.astype('float32')

            sourceclass[i, 0] = STFTclass[0]
            sourceclass[i, 1] = STFTclass[1]
            sourceclass[i, 2] = STFTclass[2]
            sourceclass[i, 3] = STFTclass[3]

        return (X_input, {"sep_sources": y, "sourceclass": sourceclass, "recov_input": X_input})
