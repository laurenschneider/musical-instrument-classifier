# Musical Instrument Classifier

<h2>Authors</h2>
Tacy Bechtel <br>
Lauren Schneider <br> <br>

Final project for Computers, Sound, & Music at Portland State University.

### Project Setup
<b>Windows: </b>
This project should be run in an Anaconda virtual environment.
Once inside the virtual environment, run the following commands:

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


### Discussion of Results
Here's where we talk about the weird dataset
