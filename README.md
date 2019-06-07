# Musical Instrument Classifier

<h2>Authors</h2>
Tacy Bechtel <br>
Lauren Schneider <br> <br>

Final project for Computers, Sound, & Music at Portland State University.

### Project Setup
<b>Windows: </b>
This project should be run in an Anaconda virtual environment.
Once inside the virtual environment, run the following commands: <br><br>
```conda install tensorflow```

```conda install -c conda-forge librosa```

```pip install -r requirements.txt```
<br>
<br>
<b>Mac OS/Linux:</b> <br> <br>
```pip install librosa```

```pip install -r requirements.txt```

This should get the environment set up properly for running the code.


### Data Preprocessing

The NSynth dataset contains over 300,000 WAV audio files.
Should you wish to re-run the preprocessing, you will need to download the NSynth dataset
and put the files into the workspace. We chose to handle the files manually because accessing
this particular dataset through TensorFlow's dataset API requires more processing power than we had available.

The processed training data file is too large to be included on GitHub. The neural network analyzes the averaged MFCCs
(Mel Frequency Cepstral Coefficients) of the sound files, found using the Librosa library.


### Challenges
Because of the size of the dataset, we decided to use only Acoustic samples in our project.
One issue we ran into was that there were two classes of acoustic instruments in the training dataset that were not represented in the test dataset.
We modified our training set to ignore those classes.

Because there are not very many good datasets for machine learning with sound files, we decided it was worth it
to work through the issues with this dataset instead of changing to another partway through or trying to create
our own.

### Discussion of Results

The neural network got to 79% accuracy in classifying the test data when run for 10 epochs. When we tried running it for more epochs,
accuracy decreased. We would have liked to have seen better results if we'd had time to play with the network's set up more.

### Lessons Learned

If we had more time, we would figure out a more specific, balanced dataset. The NSynth dataset has large disparities in
the number of files across categories, which contributed to the network overtraining as the number of epochs increased.

We spent a good amount of time trying to use too many features for each file before finally deciding on MFCCs.
Once we decided on these, the process became much simpler.