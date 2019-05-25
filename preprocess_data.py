# Import data and prepare to feed into Keras model
# extract features from wav files

import librosa
import pathlib


DATA_PATH = pathlib.Path('data')
TEST_AUDIO_PATH = DATA_PATH/'nsynth-test/audio'

# trim dataset to only one type. returns list of strings
test_acoustic = [file.name for file in TEST_AUDIO_PATH.iterdir()
                    if 'acoustic' in file.name]

for wavfile in test_acoustic:
    y, sample_rate = librosa.load(wavfile)

    # decompose to get harmonic and percussive features
    # returns numpy 2-d complex array
    stft = librosa.stft(y)
    harmonic, percuss = librosa.decompose.hpss(stft)

    # get mfccs as another feature
    # numpy 2-d float array
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)

    instrument = wavfile.split('_')     # returns a list
    instr_name = instrument[0]          # first index is the name string
