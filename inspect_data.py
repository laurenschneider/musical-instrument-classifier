"""
Extract features from wave files for inspection.
Data must be downloaded from nsynth website and placed in data dir
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pathlib


DATA_PATH = pathlib.Path('data')
TEST_AUDIO_PATH = DATA_PATH/'nsynth-test/audio'

# trim dataset to only one type
test_acoustic = [file.name for file in TEST_AUDIO_PATH.iterdir()
                    if 'acoustic' in file.name]

instrument = test_acoustic[0].split('_')
print(instrument[0])    # print first part of filename to see instrument

example_path_str = str(TEST_AUDIO_PATH/test_acoustic[0])

"""
Initial read and inspect data in time domain
"""

# get time series and sample rate
y, sample_rate = librosa.load(example_path_str)

# plot amplitude in time domain
plt.figure()
librosa.display.waveplot(y, sr=sample_rate)
plt.show()

"""
Fast Fourier Transform to convert to the frequency domain
"""

yf = np.fft.fft(y)  
N = len(y)  # number of samples
reals = int(N/2)    # only need real valued data, so take first half
yf =  yf[0:reals]

# plot FFT
plt.figure(1, figsize=(8,6))
plt.plot(yf, color='orange')
plt.xlabel('?') # TODO: what should this label be?
plt.ylabel('Amplitude')
plt.show()

"""
MFCCs
"""

mfccs = librosa.feature.mfcc(y=y, sr=sample_rate)

# visualize mfcc
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('instrument[0]')
plt.show()  # TODO: save plots in separate dir
