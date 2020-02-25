import librosa
import librosa.display
from librosa.output import write_wav
import numpy as np
import sys

def main(args):

    audio1, sr = librosa.load('output2/pred_bass1_1.wav', sr=16000)
    audio2, sr = librosa.load('output2/pred_bass1_2.wav', sr=16000)
    mix1 = np.append(audio1,audio2)

    audio3, sr = librosa.load('output2/pred_bass1_3.wav', sr=16000)
    audio4, sr = librosa.load('output2/pred_bass1_4.wav', sr=16000)
    mix2 = np.append(audio3,audio4)

    total = np.append(mix1,mix2)

    write_wav('output2/bass1.wav', total, sr)

if __name__ == "__main__":
	main(sys.argv)
