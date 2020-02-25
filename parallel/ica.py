import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import librosa
import librosa.display
from librosa.output import write_wav
from sklearn.decomposition import FastICA, PCA

def main(args):
    audio, sr = librosa.load("C:/Users/Vanessa Tan/Desktop/subjective eval/tracks/Angels In Amplifiers - I'm Alright.wav" , offset=70, duration=9, mono=True, sr=16000)
    audio = np.reshape(audio, (1, 144000))
    ica = FastICA(n_components=4)
    S_ = ica.fit_transform(audio)  # Reconstruct signals
    print(S_.shape)

if __name__ == "__main__":
	main(sys.argv)
