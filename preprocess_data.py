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
#dictionary to hold instrument names and corresponding numerical values
instrument_values = {}

counter = 0

for wavfile in test_acoustic:
    y, sample_rate = librosa.load(TEST_AUDIO_PATH/wavfile)

    # decompose to get harmonic and percussive features
    # returns numpy 2-d complex array
    """
    stft = librosa.stft(y)
    harmonic, percuss = librosa.decompose.hpss(stft)
    """

    # get mfccs as another feature
    # numpy 2-d float array
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=30)

    """
    #create single array with all values for file
    file_features = stft.flatten()
    file_features = np.append(file_features, harmonic)
    file_features = np.append(file_features, percuss)
    file_features = np.append(file_features, mfccs)
    """
    file_features = mfccs.flatten()

    #if first file, set array to this
    if not features.any():
        features = file_features
    #otherwise add this file's info as new element in the features array
    else:
        features = np.vstack((features, file_features))

    instrument = wavfile.split('_')     # returns a list
    instr_name = instrument[0]          # first index is the name string

    #add to dictionary if missing
    if instr_name not in instrument_values:
        instrument_values[instr_name] = len(instrument_values)
    #add correct value to labels that corresponds to dict entry for instrument key
    labels = np.append(labels, instrument_values[instr_name])

    if counter%100 == 0:
        print(counter, "files written")
    counter = counter + 1

print(instrument_values)


# for testing
# write arrays to files to inspect
TEST_PATH = pathlib.Path('tests')
feature_file = TEST_PATH/'features.txt'
label_file = TEST_PATH/'labels.txt'
dictionary = TEST_PATH/'dictionary.txt'


np.savetxt(feature_file, features, fmt="%s")
np.savetxt(label_file, labels, fmt="%s")
dictfile = open(dictionary,"w")
dictfile.write( str(instrument_values))
dictfile.close()
