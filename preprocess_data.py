
# Import data and prepare to feed into Keras model
# extract features from wav files

import librosa
import pathlib
import numpy as np


def get_files(test_or_train):
    data_path = pathlib.Path('data')
    folder = 'nsynth-' + test_or_train + '/audio'
    audio_path = data_path/folder

    # trim dataset to only one type. returns list of strings
    acoustic = [file.name for file in audio_path.iterdir()
                    if 'acoustic' in file.name]

    test_files(acoustic, audio_path, test_or_train)


def test_files(file_names, audio_path, test_or_train):
    features = []
    labels = np.array([])

    counter = 0

    dictionary = eval(open('dictionary.txt', 'r').read())

    for wavfile in file_names:

        y, sample_rate = librosa.load(audio_path/wavfile)

        # get mfccs
        # numpy 2-d float array
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=30)
        mfccs = np.mean(mfccs.T, axis=0)
        features.append(mfccs.flatten())

        instrument = wavfile.split('_')     # returns a list
        instr_name = instrument[0]          # first index is the name string

        # add to dictionary if missing
        """
        if instr_name not in instrument_values:
            instrument_values[instr_name] = len(instrument_values) // 2
            instrument_values[str(instrument_values[instr_name])] = instr_name
        """
        # add correct value to labels that corresponds to dict entry for instrument key
        labels = np.append(labels, dictionary[instr_name])

        if counter%100 == 0:
            print(counter, "files processed")
        counter += 1

    write_files(features, labels, test_or_train)


def write_files(features, labels, test_or_train):

    # write arrays to files to inspect
    if test_or_train == 'test':
        file_path = pathlib.Path('test')
    elif test_or_train == 'train':
        file_path = pathlib.Path('train')

    feature_file = file_path/'features.txt'
    label_file = file_path/'labels.txt'
    dictionary = file_path/'dictionary.txt'

    np.savetxt(feature_file, np.array(features), fmt="%s")
    np.savetxt(label_file, labels, fmt="%s")
    """
    dictfile = open(dictionary,"w")
    dictfile.write(str(instrument_values))
    dictfile.close()
    """


#get_files('train')
get_files('test')
