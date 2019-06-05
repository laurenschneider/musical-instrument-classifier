# unit tests for neural network

from neural_net import load_data

INPUT_DIMENSION = 30

def dataShouldMatchRequestedInputDim():
    train_features = load_data('train/features.txt')
    if INPUT_DIMENSION == train_features.shape:
        print('dimensions match')
    else:
        print('dimension mismatch')


dataShouldMatchRequestedInputDim()
