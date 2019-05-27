"""
Neural network
"""

import tensorflow as tf
from tensorflow import keras

# get data from preprocess_data.py


# build model
model = keras.models.Sequential()

# set up layers
# TODO: needs input shape
model.add(keras.layers.Dense(10, input_shape=(5190)))
model.add(keras.layers.Activation('relu'))  # can experiment with different activation fucntions

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TODO: train model

# TODO: evaluate accuracy
