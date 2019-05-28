"""
Neural network
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# get data from preprocess_data.py
features = np.loadtxt('tests/features.txt')
print(features.shape)
labels = np.loadtxt('tests/labels.txt')
print(labels.shape)

# build model
model = keras.models.Sequential()

# set up layers
# TODO: needs input shape
model.add(keras.layers.Dense(100, input_shape=(5190,)))
model.add(keras.layers.Activation('relu'))  # can experiment with different activation functions
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TODO: train model
model.fit(x=features, y=labels, epochs=20)

# TODO: create testing dataset
# TODO: evaluate accuracy
