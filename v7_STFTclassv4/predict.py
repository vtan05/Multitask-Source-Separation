# JACSNET Prediction
# Author: Vanessa H. Tan 01/25/19

# get libraries
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import librosa
import librosa.display
from librosa.output import write_wav

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import keras.backend as K
import norbert

from model import UNETmodule
from utils import spectowav

def custom_loss_wrapper_a(mask):
    def custom_loss_a(y_true, y_pred):
        mae = K.mean(K.abs(np.multiply(mask, y_pred) - y_true), axis=-1)

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(np.multiply(mask, y_pred), K.epsilon(), 1)
        KL = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

        return mae + (0.5*KL)
    return custom_loss_a

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def main(args):

    # Parameters
    seed = 3
    num_classes = 4
    num_epochs = 100
    drop_prob = 0
    learning_rate = 1e-5
    window_size = 512
    hop_size = 128
    window = np.blackman(window_size)

    # Model Architecture
    inputs = Input(shape=[257, 626, 1])

    UNETout1 = UNETmodule(inputs, num_classes, drop_prob)
    sep_sources = Activation('softmax', name='sep_sources')(UNETout1)

    UNETout2 = UNETmodule(sep_sources, 1, drop_prob)
    recov_input = Activation('sigmoid', name='recov_input')(UNETout2)

    sourceclass = GlobalAveragePooling2D()(sep_sources)
    sourceclass = Dense(1024, activation='elu')(sourceclass)
    sourceclass = Dense(1024, activation='elu')(sourceclass)
    sourceclass = Dense(1024, activation='elu')(sourceclass)
    sourceclass = Dense(1024, activation='elu')(sourceclass)
    sourceclass = Dense(num_classes)(sourceclass)
    sourceclass = Activation('sigmoid', name='sourceclass')(sourceclass)

    # Train Model Architecture
    loss_funcs = {
        "sep_sources": custom_loss_wrapper_a(mask = inputs),
        "sourceclass": binary_focal_loss(),
        "recov_input": "mean_absolute_error"
    }
    lossWeights = {"sep_sources": 10, "sourceclass": 0.01, "recov_input": 1.0}
    optimizer = optimizers.Adam(lr=learning_rate)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    model = Model(inputs=inputs, outputs=[sep_sources, sourceclass, recov_input])
    model.compile(loss=loss_funcs, optimizer=optimizer, loss_weights=lossWeights)

    # Load Model
    model.load_weights("model_weights.hdf5")

    # Predict
    audio, sr = librosa.load('D:/Data/Samples/piano.mp3', offset=60, duration=5, mono=True, sr=16000)
    write_wav('mix.wav', audio, sr, norm=True)
    orig = librosa.core.stft(audio, hop_length=hop_size, n_fft=window_size, window=window)
    magnitude, phase = librosa.magphase(orig)
    orig_norm = 2 * magnitude / np.sum(window)

    X = np.reshape(orig_norm, (1, 257, 626, 1))
    X = X.astype('float32')

    (sources, sourceclass, inp) = model.predict(X, batch_size=1)
    # sourceclass[sourceclass < 0.5] = 0
    # sourceclass[sourceclass > 0.5] = 1
    print(np.around(sourceclass, decimals = 1))

    sources_reshape = np.reshape(sources, (257, 626, 1, 4))
    orig_reshape = np.reshape(orig, (257, 626, 1))
    source_spec = norbert.wiener(sources_reshape, orig_reshape, use_softmask=True)

    inp_reshape = np.reshape(inp, (257, 626, 1, 1))
    inp = norbert.wiener(inp_reshape, orig_reshape, use_softmask=False)
    target_pred_mag_inp = np.reshape(inp, (257, 626))

    target_pred_mag_vocal = np.reshape(source_spec[:,:,:,0], (257, 626))
    target_pred_mag_bas = np.reshape(source_spec[:,:,:,1], (257, 626))
    target_pred_mag_dru = np.reshape(source_spec[:,:,:,2], (257, 626))
    target_pred_mag_oth = np.reshape(source_spec[:,:,:,3], (257, 626))

    inp_istft = librosa.core.istft(target_pred_mag_inp, hop_length=hop_size, window=window)
    write_wav('pred_input.wav', inp_istft, sr, norm=True)

    voc_istft = librosa.core.istft(target_pred_mag_vocal, hop_length=hop_size, window=window)
    write_wav('pred_vocal.wav', voc_istft, sr, norm=True)

    bas_istft = librosa.core.istft(target_pred_mag_bas, hop_length=hop_size, window=window)
    write_wav('pred_bass.wav', bas_istft, sr, norm=True)

    dru_istft = librosa.core.istft(target_pred_mag_dru, hop_length=hop_size, window=window)
    write_wav('pred_drums.wav', dru_istft, sr, norm=True)

    others_istft = librosa.core.istft(target_pred_mag_oth, hop_length=hop_size, window=window)
    write_wav('pred_others.wav', others_istft, sr, norm=True)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(target_pred_mag_inp, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title('Recovered Mixture')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(target_pred_mag_vocal, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title('Vocals')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(target_pred_mag_bas, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title('Bass')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(target_pred_mag_dru, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title('Drums')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
	main(sys.argv)