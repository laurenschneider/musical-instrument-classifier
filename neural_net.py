"""
Keras model to classify acoustic instrument sounds.

Must have run preprocess_data.py to generate keras friendly data files
before running this neural net.
"""

import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
from matplotlib import pyplot

class Neural_Net:

    def __init__(self):
        self.model = keras.models.Sequential()


    def build_model(self, category_count):
        """
        Create keras neural network model
        :param category_count: int
        """
        if category_count <= 0:
            return 0

        # set up layers
        self.model.add(keras.layers.Dense(50, input_dim=30))
        self.model.add(keras.layers.Activation('sigmoid'))
        self.model.add(keras.layers.Dense(100))
        self.model.add(keras.layers.Activation('sigmoid'))
        self.model.add(keras.layers.Dense(category_count, activation=tf.nn.sigmoid))

        self.model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        return 1

    def train(self, train_features, train_label_cats):
        """
        Train the model on training data set
        :param train_features: numpy array
        :param train_label_cats: numpy array
        """
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5',
                                                        verbose=1,
                                                        save_best_only=True)
        starttime = time.time()
        self.model.fit(x=train_features,
                        y=train_label_cats,
                        epochs=10,
                        callbacks=[checkpointer])
        endtime = time.time() - starttime
        print('one epoch in  ', endtime, 's.')


    def predict(self, test_features, test_label_cats):
        """
        Test the model to predict instruments
        :param test_features: numpy array
        :param test_label_cats: numpy array
        """
        # create testing dataset
        self.model.evaluate(test_features, test_label_cats)

        # evaluate accuracy
        predictions = self.model.predict(test_features)
        flat_predictions = np.argmax(predictions, axis=1)
        print(flat_predictions)
        print(predictions.shape)
        flat_actuals = np.argmax(test_label_cats, axis=1)
        print(flat_actuals)
        self.conf_matrix = confusion_matrix(flat_actuals, flat_predictions)


    def plot_heatmap(self, labels):
        """
        Display results of prediction
        :param labels: array
        """
        seaborn.heatmap(self.conf_matrix,
                        cmap='Blues',
                        annot=True,
                        fmt='d',
                        xticklabels=labels,
                        yticklabels=labels)
        pyplot.xlabel('Actual')
        pyplot.ylabel('Predicted')
        pyplot.title('Classification of Instruments from wav files')
        pyplot.savefig('matrix.png')
        pyplot.show()
