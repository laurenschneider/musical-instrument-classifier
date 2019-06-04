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


def load_data(filepath):
    """
    Load preprocessed data to run through keras model.
    :param filepath: String
    :returns: numpy array
    """
    starttime = time.time()
    # get data from preprocessed files
    data = np.loadtxt(filepath)
    endtime = time.time() - starttime
    print('loaded data in ', endtime, 's.')
    return data


# get data

train_labels = load_data('train/labels.txt')
train_features = load_data('train/features.txt')
test_labels = load_data('test/labels.txt')
test_features = load_data('test/features.txt')


dictfile = open('dictionary.txt', 'r')
instr_dict = eval(dictfile.read())
category_count = len(instr_dict)
train_label_cats = keras.utils.to_categorical(train_labels, num_classes=category_count)
test_label_cats = keras.utils.to_categorical(test_labels, num_classes=category_count)

# build model

model = keras.models.Sequential()

# set up layers
model.add(keras.layers.Dense(50, input_dim=30))
model.add(keras.layers.Activation('sigmoid'))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Activation('sigmoid'))
model.add(keras.layers.Dense(category_count, activation=tf.nn.sigmoid))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model

# use checkpointer to hold weights to prevent overtraining
checkpointer = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
starttime = time.time()
model.fit(x=train_features, y=train_label_cats, epochs=10, callbacks=[checkpointer])
endtime = time.time() - starttime
print('one epoch in  ', endtime, 's.')

# create testing dataset
model.evaluate(test_features, test_label_cats)

# evaluate accuracy
predictions = model.predict(test_features)
flat_predictions = np.argmax(predictions, axis=1)
print(flat_predictions)
print(predictions.shape)
flat_actuals = np.argmax(test_label_cats, axis=1)
print(flat_actuals)
conf_matrix = confusion_matrix(flat_actuals, flat_predictions)
print(conf_matrix.shape)

all_labels = instr_dict.keys()
seaborn.heatmap(conf_matrix, cmap='Blues', annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels)
pyplot.xlabel('Actual')
pyplot.ylabel('Predicted')
pyplot.title('Classification of Instruments from wav files')
pyplot.savefig('matrix.png')
pyplot.show()
