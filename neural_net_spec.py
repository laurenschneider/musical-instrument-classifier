# unit tests for neural network

from neural_net import Neural_Net

TEST_NN = Neural_Net()

result = TEST_NN.build_model(0)
print("category count passed") if result==0 else print("category count failed")

result = TEST_NN.build_model(-2)
print("category count passed") if result==0 else print("category count failed")

result = TEST_NN.build_model(10)
print("category count passed") if result==1 else print("category count failed")
