"""
Train a keras neural network to predict acoustic instrument sounds.
"""

import time
import tensorflow as tf
from tensorflow import keras
from neural_net import Neural_Net
import numpy as np


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

# import data
train_labels = load_data('train/labels.txt')
train_features = load_data('train/features.txt')
test_labels = load_data('test/labels.txt')
test_features = load_data('test/features.txt')

dictfile = open('dictionary.txt', 'r')
instr_dict = eval(dictfile.read())
category_count = len(instr_dict)
train_label_cats = keras.utils.to_categorical(train_labels, num_classes=category_count)
test_label_cats = keras.utils.to_categorical(test_labels, num_classes=category_count)
all_labels = instr_dict.keys()

# run model and see results
model = Neural_Net()
model.build_model(category_count)
model.train(train_features, train_label_cats)
model.predict(test_features, test_label_cats)
model.plot_heatmap(all_labels)
