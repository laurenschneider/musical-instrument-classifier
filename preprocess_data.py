# Import data and prepare to feed into Keras model

import tensorflow_datasets as tfds

# get dataset

data = tfds.load("nsynth")
