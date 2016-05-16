clear;clc;
X_train = loadMNISTImages('train-images-idx3-ubyte')'; % (60,000 x 784)
Y_train = loadMNISTLabels('train-labels-idx1-ubyte'); % (60,000 x 1)
X_test = loadMNISTImages('t10k-images-idx3-ubyte')'; % (10,000 x 784)
Y_test = loadMNISTLabels('t10k-labels-idx1-ubyte'); % (10,000 x 1)
save('data_MNIST_original_minist_60k_10k_split_train_test')