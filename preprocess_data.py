# Import data and prepare to feed into Keras model
# extract features from wav files

import librosa
import pathlib
import numpy as np

DATA_PATH = pathlib.Path('data')
TEST_AUDIO_PATH = DATA_PATH/'nsynth-test/audio'

# trim dataset to only one type. returns list of strings
test_acoustic = [file.name for file in TEST_AUDIO_PATH.iterdir()
                    if 'acoustic' in file.name]

features = np.array([[]])
labels = np.array([])

counter = 0

for wavfile in test_acoustic:
    y, sample_rate = librosa.load(TEST_AUDIO_PATH/wavfile)

    # decompose to get harmonic and percussive features
    # returns numpy 2-d complex array
    #print(wavfile)
    stft = librosa.stft(y)
    #print('stft:  ', stft)
    harmonic, percuss = librosa.decompose.hpss(stft)
    #print('harmonic: ', harmonic)
    #print('percuss: ', percuss)

    # get mfccs as another feature
    # numpy 2-d float array
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=30)
    #print('mfccs: ', mfccs.shape)
    #print('mfccs: ', mfccs)
    #TODO Each file's info should be compressed to one line
    #(current is 30x173)
    if not features.any():
        features = mfccs
    else:
        features = np.vstack((features, mfccs))
    print('features: ', features.shape)

    instrument = wavfile.split('_')     # returns a list
    instr_name = instrument[0]          # first index is the name string
    labels = np.append(labels, [instr_name])

    if counter%100 == 0:
        print(counter, "files written")
    counter = counter + 1


# for testing
# write arrays to files to inspect
TEST_PATH = pathlib.Path('tests')
feature_file = TEST_PATH/'features.txt'
label_file = TEST_PATH/'labels.txt'

np.savetxt(feature_file, features, fmt="%s")
np.savetxt(label_file, labels, fmt="%s")
