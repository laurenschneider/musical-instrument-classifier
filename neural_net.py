"""
Neural network to classify acoustic instrument sounds.

Must have run preprocess_data.py to generate keras friendly data files
before running this neural net.
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import seaborn
from matplotlib import pyplot

starttime = time.time()
# get data from preprocessed files
train_features = np.loadtxt('train/features.txt')
endtime = time.time() - starttime
print('loaded training data in ', endtime, 's.')
#features = np.loadtxt('tests/features.txt')

starttime = time.time()
test_features = np.loadtxt('test/features.txt')
endtime = time.time() - starttime
print('loaded test data in ', endtime, 's.')
# print(features.shape)

starttime = time.time()
train_labels = np.loadtxt('train/labels.txt')
endtime = time.time() - starttime
print('loaded training labels in ', endtime, 's.')

#labels = np.loadtxt('tests/labels.txt')
starttime = time.time()
test_labels = np.loadtxt('test/labels.txt')
endtime = time.time() - starttime
print('loaded test labels in ', endtime, 's.')
starttime = time.time()

# print(labels.shape)
train_dictfile = open('train/dictionary.txt', 'r')
test_dictfile = open('test/dictionary.txt', 'r')

train_instruments = eval(train_dictfile.read())
"""
test_instruments = eval(test_dictfile.read())
if train_dictfile != test_dictfile:
    print("dictionary mismatch")
    exit(1)
"""
category_count = len(train_instruments)
train_label_cats = keras.utils.to_categorical(train_labels, num_classes=category_count)
test_label_cats = keras.utils.to_categorical(test_labels, num_classes=category_count)
#print(label_cats.shape)
# build model

all_labels = train_instruments.keys()


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

# train model
starttime = time.time()
model.fit(x=train_features, y=train_label_cats, epochs=1)
endtime = time.time() - starttime
print('one epoch in  ', endtime, 's.')

# TODO: create testing dataset
model.evaluate(test_features, test_label_cats)
# TODO: evaluate accuracy
predictions = model.predict(test_features)
flat_predictions = np.argmax(predictions, axis=1)
print(flat_predictions)
print(predictions.shape)
flat_actuals = np.argmax(test_label_cats, axis=1)
print(flat_actuals)
conf_matrix = confusion_matrix(flat_actuals, flat_predictions)
print(conf_matrix.shape)

seaborn.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels)
pyplot.xlabel('Actual')
pyplot.ylabel('Predicted')
pyplot.title('Classification of Instruments from wav files')
pyplot.savefig('matrix.png')
pyplot.show()
