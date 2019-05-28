# Import data and prepare to feed into Keras model
# extract features from wav files

import librosa
import pathlib
import numpy as np
# import time
import tensorflow_datasets as tfds


# start_time = time.time()


def get_train():
    all_data = tfds.load('nsynth', with_info=False)
    train_data = all_data['train']
    print(train_data.shape)
    train_acoustic = [file.name for file in train_data.iterdir()
                      if 'acoustic' in file.name]
    train_files(train_acoustic)


def train_files(train_acoustic):
    features = []
    labels = np.array([])
    instrument_values = {}

    write_files(features, labels, instrument_values, 'train')


def get_tests():
    data_path = pathlib.Path('data')
    test_audio_path = data_path/'nsynth-test/audio'

    # trim dataset to only one type. returns list of strings
    test_acoustic = [file.name for file in test_audio_path.iterdir()
                    if 'acoustic' in file.name]

    test_files(test_acoustic, test_audio_path)


def test_files(file_names, audio_path):
    features = []
    labels = np.array([])
    instrument_values = {}

    counter = 0

    for wavfile in file_names:

        y, sample_rate = librosa.load(audio_path/wavfile)

        # get mfccs as another feature
        # numpy 2-d float array
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=30)

        features.append(mfccs.flatten())

        instrument = wavfile.split('_')     # returns a list
        instr_name = instrument[0]          # first index is the name string

        # add to dictionary if missing
        if instr_name not in instrument_values:
            instrument_values[instr_name] = len(instrument_values)

        # add correct value to labels that corresponds to dict entry for instrument key
        labels = np.append(labels, instrument_values[instr_name])

        if counter%100 == 0:
            print(counter, "files written")
        counter = counter + 1

    write_files(features, labels, instrument_values, 'test')


def write_files(features, labels, instrument_values, test_or_train):

    # write arrays to files to inspect
    test_path = pathlib.Path(test_or_train)
    feature_file = test_path/'features.txt'
    label_file = test_path/'labels.txt'
    dictionary = test_path/'dictionary.txt'

    np.savetxt(feature_file, np.array(features), fmt="%s")
    np.savetxt(label_file, labels, fmt="%s")
    dictfile = open(dictionary,"w")
    dictfile.write(str(instrument_values))
    dictfile.close()
    # print("time elapsed: {:.2f}s".format(time.time() - start_time))


get_train()


