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

    conv1 = Conv2D(32, 3, dilation_rate=1, padding='same')(inputs)
    conv1 = ELU()(conv1)

    conv2 = Conv2D(32, 3, dilation_rate=2, padding='same')(conv1)
    conv2 = ELU()(conv2)

    conv3 = Conv2D(32, 3, dilation_rate=3, padding='same')(conv2)
    conv3 = ELU()(conv3)

    conv4 = Conv2D(64, 3, dilation_rate=5, padding='same')(conv3)
    conv4 = ELU()(conv4)

    conv5 = Conv2D(128, 3, dilation_rate=8, padding='same')(conv4)
    conv5 = ELU()(conv5)

    deconv6 = Conv2DTranspose(64, 3, dilation_rate=5, padding='same')(conv5)
    deconv6 = ELU()(deconv6)

    deconv7 = Concatenate(axis=3)([deconv6, conv4])
    deconv7 = Conv2DTranspose(32, 3, dilation_rate=3, padding='same')(deconv7)
    deconv7 = ELU()(deconv7)

    deconv8 = Concatenate(axis=3)([deconv7, conv3])
    deconv8 = Conv2DTranspose(32, 3, dilation_rate=2, padding='same')(deconv8)
    deconv8 = ELU()(deconv8)

    deconv9 = Concatenate(axis=3)([deconv8, conv2])
    deconv9 = Conv2DTranspose(32, 3, dilation_rate=1, padding='same')(deconv9)
    deconv9 = ELU()(deconv9)

    deconv10 = Concatenate(axis=3)([deconv9, conv1])
    deconv10 = Conv2DTranspose(out_filter, 3, dilation_rate=1, padding='same')(deconv10)

    return deconv10
