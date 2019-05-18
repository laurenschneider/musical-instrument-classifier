# Import data and prepare to feed into Keras model
# extract features from wav files

import librosa

data = []   # list of .wav file path names

for wavfile in data:

    y, sample_rate = librosa.load(wavfile)

    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)
