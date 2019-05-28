"""
Neural network
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# get data from preprocess_data.py
features = np.loadtxt('tests/features.txt')
# print(features.shape)
labels = np.loadtxt('tests/labels.txt')
# print(labels.shape)
dictfile = open('tests/dictionary.txt', 'r')
instruments = eval(dictfile.read())
category_count = len(instruments)

label_cats = keras.utils.to_categorical(labels, num_classes=category_count)
print(label_cats.shape)
# build model
model = keras.models.Sequential()

# set up layers
model.add(keras.layers.Dense(50, input_dim=5190))
model.add(keras.layers.Activation('sigmoid'))
#model.add(keras.layers.Activation('relu'))  # can experiment with different activation functions
model.add(keras.layers.Dense(100))
model.add(keras.layers.Activation('sigmoid'))
model.add(keras.layers.Dense(category_count, activation=tf.nn.sigmoid))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TODO: train model
model.fit(x=features, y=label_cats, epochs=50)

# TODO: create testing dataset
# TODO: evaluate accuracy
