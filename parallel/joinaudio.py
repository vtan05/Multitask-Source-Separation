import librosa
import librosa.display
from librosa.output import write_wav
import numpy as np
import sys

def main(args):

    audio1, sr = librosa.load("defense/Angels In Amplifiers - I'm Alright/pred_others1_1.wav", sr=16000)
    audio2, sr = librosa.load("defense/Angels In Amplifiers - I'm Alright/pred_others1_2.wav", sr=16000)
    mix1 = np.append(audio1,audio2)

    audio3, sr = librosa.load("defense/Angels In Amplifiers - I'm Alright/pred_others1_3.wav", sr=16000)
    audio4, sr = librosa.load("defense/Angels In Amplifiers - I'm Alright/pred_others1_4.wav", sr=16000)
    mix2 = np.append(audio3,audio4)

    total = np.append(mix1,mix2)

    write_wav("defense/Angels In Amplifiers - I'm Alright/others1.wav", total, sr)

if __name__ == "__main__":
	main(sys.argv)
