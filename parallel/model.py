import sys
import numpy as np
import pandas as pd
import scipy
import librosa
import sklearn

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import keras.backend as K

def UNETmodule(inputs, out_filter, drop_prob):

    conv1 = Conv2D(16, 3, dilation_rate=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(32, 3, dilation_rate=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(64, 3, dilation_rate=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(128, 3, dilation_rate=5, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    deconv5 = Conv2DTranspose(64, 3, dilation_rate=3, padding='same')(conv4)
    deconv5 = BatchNormalization()(deconv5)
    # deconv5 = Dropout(drop_prob)(deconv5)
    deconv5 = LeakyReLU(alpha=0.2)(deconv5)

    deconv6 = Concatenate(axis=3)([deconv5, conv3])
    deconv6 = Conv2DTranspose(32, 3, dilation_rate=2, padding='same')(deconv6)
    deconv6 = BatchNormalization()(deconv6)
    # deconv6 = Dropout(drop_prob)(deconv6)
    deconv6 = LeakyReLU(alpha=0.2)(deconv6)

    deconv7 = Concatenate(axis=3)([deconv6, conv2])
    deconv7 = Conv2DTranspose(16, 3, dilation_rate=1, padding='same')(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    # deconv7 = Dropout(drop_prob)(deconv7)
    deconv7 = LeakyReLU(alpha=0.2)(deconv7)

    deconv8 = Concatenate(axis=3)([deconv7, conv1])
    deconv8 = Conv2DTranspose(out_filter, 3, dilation_rate=1, padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    # deconv8 = Dropout(drop_prob)(deconv8)

    return deconv8

def RECOVmodule(inputs, out_filter, drop_prob):

    conv1 = Conv2D(32, 3, dilation_rate=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(64, 3, dilation_rate=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(128, 3, dilation_rate=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    deconv4 = Conv2DTranspose(64, 3, dilation_rate=2, padding='same')(conv3)
    deconv4 = BatchNormalization()(deconv4)
    # deconv4 = Dropout(drop_prob)(deconv4)
    deconv4 = LeakyReLU(alpha=0.2)(deconv4)

    deconv5 = Concatenate(axis=3)([deconv4, conv2])
    deconv5 = Conv2DTranspose(32, 3, dilation_rate=1, padding='same')(deconv5)
    deconv5 = BatchNormalization()(deconv5)
    # deconv5 = Dropout(drop_prob)(deconv5)
    deconv5 = LeakyReLU(alpha=0.2)(deconv5)

    deconv6 = Concatenate(axis=3)([deconv5, conv1])
    deconv6 = Conv2DTranspose(out_filter, 3, dilation_rate=1, padding='same')(deconv6)
    deconv6 = BatchNormalization()(deconv6)

    return deconv6
