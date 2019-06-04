# Testing of the data preprocessing step

import numpy as np

# check data files contain same labels as dictionary

def trainDataShouldContainSameLabelsAsDict():
    dictfile = open('dictionary.txt', 'r')
    dictionary = eval(dictfile.read())

    labels = np.loadtxt('train/labels.txt')
    labels = labels.astype(int)

    for instrument in dictionary:
        if dictionary[instrument] not in labels:
            print(dictionary[instrument], instrument, "not found in train labels")


def testDataShouldContainSameLabelsAsDict():
    dictfile = open('dictionary.txt', 'r')
    dictionary = eval(dictfile.read())

    labels = np.loadtxt('test/labels.txt')
    labels = labels.astype(int)

    for instrument in dictionary:
        if dictionary[instrument] not in labels:
            print(dictionary[instrument], instrument, "not found in test labels")


trainDataShouldContainSameLabelsAsDict()
testDataShouldContainSameLabelsAsDict()
