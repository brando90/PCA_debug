clear;clc;
load('data_MNIST_original_minist_60k_10k_split_train_test');
[N_train, D] = size(X_train) % (N_train x D)
[N_test, D] = size(X_test) % (N_test x D)
%% preparing model
K = 10;
[coeff, ~, ~, ~, ~, mu] = pca(X_train); % (D x R) = Rows of X_train correspond to observations and columns to variables. 
%% Learn PCA
X_train = X_train'; % (D x N) = (N_train x D)'
X_test = X_test'; % (D x N) = (N_test x D)'
U = coeff(:,1:K); % (D x K) = K pca's of dimension D
X_tilde_train = (U * U' * X_train); % (D x N_train)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
X_tilde_test = (U * U' * X_test); % (D x N_test)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
train_error_PCA = (1/N_train)*norm( X_tilde_train - X_train ,'fro')^2
test_error_PCA = (1/N_test)*norm( X_tilde_test - X_test ,'fro')^2