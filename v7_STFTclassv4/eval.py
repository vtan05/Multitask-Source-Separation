# JACSNET Evaluation
# Author: Vanessa H. Tan 04.11.19

# get libraries
import sys
import numpy as np
import pandas as pd
import scipy
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

import librosa
import librosa.display

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import keras.backend as K

import museval
import musdb
import norbert

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, f1_score, auc, average_precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model import UNETmodule
from utils import checkNoise

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

def mulphase(inp, phase):
    voc_spec = np.abs(inp)
    voc_spec = np.multiply(voc_spec, phase)
    return voc_spec

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)

def eval_metrics(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_res = roc_auc_score_FIXED(y_true, y_pred)
    return auc_res, eer

def plot_confusion_matrix(cm, classes, title, ax):

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks), ax.xaxis.set_ticklabels(classes)
    ax.set_yticks(tick_marks), ax.yaxis.set_ticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    ax.set_title(title)
    ax.grid(False)

def plot_multiclass_confusion_matrix(y_true, y_pred, label_to_class, save_plot=False):
    fig, axes = plt.subplots(int(np.ceil(len(label_to_class) / 2)), 2, figsize=(5, 5))
    axes = axes.flatten()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        f1 = 2 * tp / (2 * tp + fp + fn + sys.float_info.epsilon)
        recall = tp / (tp + fn + sys.float_info.epsilon)
        precision = tp / (tp + fp + sys.float_info.epsilon)
        plot_confusion_matrix(
            np.array([[tp, fn], [fp, tn]]),
            classes=['+', '-'],
            title=f'Label: {label_to_class[i]}\nf1={f1:.5f}\nrecall={recall:.5f}\nprecision={precision:.5f}',
            ax=axes[i]
        )
        plt.tight_layout()
    plt.show()
    if save_plot:
        plt.savefig('confusion_matrices.png', dpi=50)

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

    # initiate musdb
    mus = musdb.DB(root_dir="D:/Data/Musdb18")

    # load the testing tracks
    tracks = mus.load_mus_tracks(subsets=['test'])

    SDR_results = []
    SIR_results = []
    ISR_results = []
    SAR_results = []

    classresults = []
    classresults_hat = []
    labelresults = []

    duration = 5
    for track in tracks:
        print(track.name)

        audio, sr = librosa.load(str('D:/Data/Musdb18/mixture/test/' + track.name + '.wav'))
        dur = librosa.get_duration(y=audio, sr=sr)

        for i in range(0, int(dur - duration), duration):
            labelclass = [0, 0, 0, 0]

            # STFT Mixture
            mixture, sr = librosa.load(str('D:/Data/Musdb18/mixture/test/' + track.name + '.wav'), offset=i, duration=duration, mono=True, sr=16000)
            orig = librosa.core.stft(mixture, hop_length=hop_size, n_fft=window_size, window=window)
            magnitude, phase = librosa.magphase(orig)
            orig_norm = 2 * magnitude / np.sum(window)

            X = np.reshape(orig_norm, (1, 257, 626, 1))
            X = X.astype('float32')

            (sources, sourceclass, inp) = model.predict(X, batch_size=1)
            classresults.append(sourceclass)
            sourceclass_hat = sourceclass
            sourceclass_hat[sourceclass_hat <= 0.5] = 0
            sourceclass_hat[sourceclass_hat > 0.5] = 1
            classresults_hat.append(sourceclass_hat)

            sources_reshape = np.reshape(sources, (257, 626, 1, 4))
            orig_reshape = np.reshape(orig, (257, 626, 1))
            source_spec = norbert.wiener(sources_reshape, orig_reshape, use_softmask=True)

            inp_reshape = np.reshape(inp, (257, 626, 1, 1))
            inp = norbert.wiener(inp_reshape, orig_reshape, use_softmask=False)
            inp_spec = np.reshape(inp, (257, 626))
            #inp_spec = mulphase(target_pred_mag_inp, phase)

            voc_spec = np.reshape(source_spec[:,:,:,0], (257, 626))
            bas_spec = np.reshape(source_spec[:,:,:,1], (257, 626))
            dru_spec = np.reshape(source_spec[:,:,:,2], (257, 626))
            oth_spec = np.reshape(source_spec[:,:,:,3], (257, 626))

            # Get ground truth
            gt_voc, sr = librosa.load(str('D:/Data/Musdb18/groundtruth/test/' + track.name + '/vocals.wav'), offset=i, duration=5, mono=True, sr=16000)
            if ((checkNoise(gt_voc) >= 0.05)): labelclass[0] = 1
            # gt_voc_down = scipy.signal.decimate(gt_voc, 2)
            gt_voc_final = np.reshape(gt_voc, (1, gt_voc.shape[0], 1))

            gt_bas, sr = librosa.load(str('D:/Data/Musdb18/groundtruth/test/' + track.name + '/bass.wav'), offset=i, duration=5, mono=True, sr=16000)
            if ((checkNoise(gt_bas) >= 0.05)): labelclass[1] = 1
            # gt_bas_down = scipy.signal.decimate(gt_bas, 2)
            gt_bas_final = np.reshape(gt_bas, (1, gt_bas.shape[0], 1))

            gt_dru, sr = librosa.load(str('D:/Data/Musdb18/groundtruth/test/' + track.name + '/drums.wav'), offset=i, duration=5, mono=True, sr=16000)
            if ((checkNoise(gt_dru) >= 0.05)): labelclass[2] = 1
            # gt_dru_down = scipy.signal.decimate(gt_dru, 2)
            gt_dru_final = np.reshape(gt_dru, (1, gt_dru.shape[0], 1))

            gt_others, sr = librosa.load(str('D:/Data/Musdb18/groundtruth/test/' + track.name + '/other.wav'), offset=i, duration=5, mono=True, sr=16000)
            if ((checkNoise(gt_others) >= 0.05)): labelclass[3] = 1
            # gt_others_down = scipy.signal.decimate(gt_others, 2)
            gt_others_final = np.reshape(gt_others, (1, gt_others.shape[0], 1))

            gt_inp_final = np.reshape(mixture, (1, mixture.shape[0], 1))

            labelresults.append(labelclass)

            # Get predictions
            vocals = librosa.core.istft(voc_spec, hop_length=hop_size, length=gt_voc_final.shape[1], window=window)
            # vocals = scipy.signal.wiener(vocals)
            vocals = np.reshape(vocals, (1, vocals.shape[0], 1))

            bass = librosa.core.istft(bas_spec, hop_length=hop_size, length=gt_bas_final.shape[1], window=window)
            # bass = scipy.signal.wiener(bass)
            bass = np.reshape(bass, (1, bass.shape[0], 1))

            drums = librosa.core.istft(dru_spec, hop_length=hop_size, length=gt_dru_final.shape[1], window=window)
            # drums = scipy.signal.wiener(drums)
            drums = np.reshape(drums, (1, drums.shape[0], 1))

            others = librosa.core.istft(oth_spec, hop_length=hop_size, length=gt_others_final.shape[1], window=window)
            # others = scipy.signal.wiener(others)
            others = np.reshape(others, (1, others.shape[0], 1))

            recov = librosa.core.istft(inp_spec, hop_length=hop_size, length=gt_inp_final.shape[1], window=window)
            recov = scipy.signal.wiener(recov)
            recov = np.reshape(recov, (1, recov.shape[0], 1))

            all_zeros = np.all(gt_voc<=0) or np.all(gt_bas<=0) or np.all(gt_dru<=0) or np.all(gt_others<=0) or np.all(vocals<=0) or np.all(bass<=0) or np.all(drums<=0) or np.all(others<=0)
            # print(all_zeros)

            if all_zeros == False:
                noise_thresh = checkNoise(gt_voc)

                if noise_thresh >= 0.05:
                    noise_thresh = checkNoise(gt_bas)

                    if noise_thresh >= 0.05:
                        noise_thresh = checkNoise(gt_dru)

                        if noise_thresh >= 0.05:
                            # Evaluate
                            REF = np.concatenate((gt_voc_final, gt_bas_final, gt_dru_final, gt_others_final, gt_inp_final), axis=0)
                            EST = np.concatenate((vocals, bass, drums, others, recov), axis=0)

                            [SDR, ISR, SIR, SAR] = museval.evaluate(REF, EST, win=55125,  hop=55125)
                            SDR_results.append(SDR)
                            ISR_results.append(ISR)
                            SIR_results.append(SIR)
                            SAR_results.append(SAR)

    y_true = np.array(labelresults, dtype=float)
    y_pred = np.array(classresults, dtype=float)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[2]))
    y_pred_hat = np.array(classresults_hat, dtype=float)
    y_pred_hat = np.reshape(y_pred_hat, (y_pred_hat.shape[0], y_pred_hat.shape[2]))

    auc_voc, eer_voc = eval_metrics(y_true[:,0], y_pred[:,0])
    auc_bas, eer_bas  = eval_metrics(y_true[:,1], y_pred[:,1])
    auc_dru, eer_dru  = eval_metrics(y_true[:,2], y_pred[:,2])
    auc_oth, eer_oth  = eval_metrics(y_true[:,3], y_pred[:,3])

    target_names = ['Vocals', 'Bass', 'Drums', 'Others']
    plot_multiclass_confusion_matrix(y_true, y_pred_hat, target_names, save_plot=False)

    SDR_array = np.array(SDR_results)
    SDR_array = np.reshape(SDR_array, (SDR_array.shape[0], SDR_array.shape[1]))
    SDR_df = pd.DataFrame(SDR_array)
    SDR_df.to_csv('SDR.csv')

    ISR_array = np.array(ISR_results)
    ISR_array = np.reshape(ISR_array, (ISR_array.shape[0], ISR_array.shape[1]))
    ISR_df = pd.DataFrame(ISR_array)
    ISR_df.to_csv('ISR.csv')

    SIR_array = np.array(SIR_results)
    SIR_array = np.reshape(SIR_array, (SIR_array.shape[0], SIR_array.shape[1]))
    SIR_df = pd.DataFrame(SIR_array)
    SIR_df.to_csv('SIR.csv')

    SAR_array = np.array(SAR_results)
    SAR_array = np.reshape(SAR_array, (SAR_array.shape[0], SAR_array.shape[1]))
    SAR_df = pd.DataFrame(SAR_array)
    SAR_df.to_csv('SAR.csv')

    print("Vocals: AUC = " + str(auc_voc) + " | EER = " + str(eer_voc) + " | SDR = " + str(np.round(np.nanmedian(SDR_array[:,0]), decimals=3)))
    print("Bass: AUC = " + str(auc_bas) + " | EER = " + str(eer_bas) + " | SDR = " + str(np.round(np.nanmedian(SDR_array[:,1]), decimals=3)))
    print("Drums: AUC = " + str(auc_dru) + " | EER = " + str(eer_dru) + " | SDR = " + str(np.round(np.nanmedian(SDR_array[:,2]), decimals=3)))
    print("Others: AUC = " + str(auc_oth) + " | EER = " + str(eer_oth) + " | SDR = " + str(np.round(np.nanmedian(SDR_array[:,3]), decimals=3)))
    print("SDR Recovered = " + str(np.round(np.nanmedian(SDR_array[:,4]), decimals=3)))

if __name__ == "__main__":
	main(sys.argv)
