"""
Extract features from wave files for inspection.
Data must be downloaded from nsynth website and placed in data dir
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib


DATA_PATH = pathlib.Path('data')
TEST_AUDIO_PATH = DATA_PATH/'nsynth-test/audio'

# trim dataset to only one type
test_acoustic = [file.name for file in TEST_AUDIO_PATH.iterdir()
                    if 'acoustic' in file.name]

instrument = test_acoustic[0].split('_')
print(instrument[0])    # print first part of filename to see instrument

example_path_str = str(TEST_AUDIO_PATH/test_acoustic[0])

# get time series and sample rate
y, sample_rate = librosa.load(example_path_str)

print("y: ", y)
print("sample rate: ", sample_rate)

mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)
print(mfccs)

# visualize mfcc
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('instrument[0]')
plt.show()  # TODO: save plots in separate dir 
