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

instrument = test_acoustic[1200].split('_')
print(instrument[0])    # print first part of filename to see instrument

example_path_str = str(TEST_AUDIO_PATH/test_acoustic[1200])

"""
Initial read and inspect data in time domain
"""

# get time series and sample rate
y, sample_rate = librosa.load(example_path_str)
"""
# plot amplitude in time domain
plt.figure()
librosa.display.waveplot(y, sr=sample_rate)
plt.show()
"""
"""
Fast Fourier Transform to convert to the frequency domain
"""

yf = np.fft.fft(y)
N = len(y)  # number of samples
reals = int(N/2)    # only need real valued data, so take first half
yf =  yf[0:reals]
"""
# plot FFT
plt.figure(1, figsize=(8,6))
plt.plot(yf, color='orange')
plt.xlabel('?') # TODO: what should this label be?
plt.ylabel('Amplitude')
plt.show()
"""
"""
MFCCs - Mel-frequency cepstral coefficient, like timbre
"""

mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=30)
print("mfcc type")
print(type(mfccs[0][0]))

# get mean
#avg = np.mean(mfccs.T, axis=0)
#print("average")
#print(avg)

"""
# visualize mfcc
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('instrument[0]')
plt.show()  # TODO: save plots in separate dir
"""


"""
Decomposition
"""

stft = librosa.stft(y)    # apply short time fourier transform
harmonic, percuss = librosa.decompose.hpss(stft)  # get harmonic and percussive components
print("harmonic type")
print(type(harmonic[0][0]))
plt.figure()

plt.subplot(3,1,1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft),
ref=np.max), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Full Power Spectogram')

plt.subplot(3,1,2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(harmonic),
                                                 ref=np.max), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic Power Spectogram')

plt.subplot(3,1,3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(percuss),
                                                 ref=np.max), y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Percussive Power Spectogram')
#plt.show()
plt.savefig('plots/' + instrument[0] +'.png')
