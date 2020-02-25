# JACSNet
# Author: Vanessa H. Tan 04.11.19

# get libraries
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf

import librosa
import librosa.display

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import keras.backend as K
from keras.utils import plot_model

import pydotplus
from keras.utils.vis_utils import model_to_dot
from STFTgenerator import Generator
from model import UNETmodule, RECOVmodule

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def custom_loss_wrapper_a(mask):
    def custom_loss_a(y_true, y_pred):
        mae = K.mean(K.abs(np.multiply(mask, y_pred) - y_true), axis=-1)

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(np.multiply(mask, y_pred), K.epsilon(), 1)
        KL = K.sum(y_true * K.log(y_true / y_pred), axis=-1)

        return mae + (0.5 * KL)
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

def main():

    # Parameters
    seed = 3
    num_epochs = 100
    learning_rate = 1e-4
    num_classes = 4
    drop_prob = 0.8

    train_params = {'dim': (1025,94),
          'batch_size': 2,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': True}

    val_params = {'dim': (1025,94),
          'batch_size': 2,
          'n_classes': 4,
          'n_channels': 1,
          'shuffle': False}

    # Generators
    training_generator = Generator(mode="train", **train_params)
    valid_generator = Generator(mode="valid", **val_params)

    # Model Architecture
    inputs = Input(shape=[1025, 94, 1])

    recov_input = UNETmodule(inputs, 1, drop_prob)
    recov_input = LeakyReLU(alpha=0.2, name='recov_input')(recov_input)

    UNET = UNETmodule(recov_input, num_classes, drop_prob)
    sep_sources = Activation('softmax', name='sep_sources')(UNET)

    sourceclass = GlobalAveragePooling2D()(UNET)
    sourceclass = Dense(128, activation='relu')(sourceclass)
    sourceclass = Dense(128, activation='relu')(sourceclass)
    sourceclass = Dense(128, activation='relu')(sourceclass)
    sourceclass = Dense(num_classes)(sourceclass)
    sourceclass = Activation('sigmoid', name='sourceclass')(sourceclass)

    # Train Model Architecture
    loss_funcs = {
        "sep_sources": custom_loss_wrapper_a(mask = inputs),
        "sourceclass": binary_focal_loss(),
        "recov_input": "binary_crossentropy"
    }
    lossWeights = {"sep_sources": 10, "sourceclass": 0.01, "recov_input": 0.5}
    optimizer = optimizers.Adam(lr=learning_rate)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    model = Model(inputs=inputs, outputs=[sep_sources, sourceclass, recov_input])
    model.compile(loss=loss_funcs, optimizer=optimizer, loss_weights=lossWeights)

    # print(model.summary())

    checkpointer = ModelCheckpoint(filepath="model_weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit_generator(generator=training_generator, validation_data=valid_generator, workers=16,
                    callbacks=[early_stop, checkpointer, reduce_lr],
                    verbose=1,
                    epochs=num_epochs,
                    shuffle=True)

    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "sep_sources_loss", "sourceclass_loss", "recov_input_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

    # loop over the loss names
    for (i, l) in enumerate(lossNames):
	       # plot the loss for both the training and validation data
	       title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	       ax[i].set_title(title)
	       ax[i].set_xlabel("Epoch")
	       ax[i].set_ylabel("Loss")
	       ax[i].plot(history.history[l], label=l)
	       ax[i].plot(history.history["val_" + l], label="val_" + l)
	       ax[i].legend()

    # save the losses figure and create a new figure for the accuracies
    plt.tight_layout()
    plt.savefig("losses.png")
    plt.close()

if __name__ == '__main__':
    main()
