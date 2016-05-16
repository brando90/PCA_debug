clear;clc;
%load('data_MNIST_original_minist_60k_10k_split_train_test');
load('mnist_background_random_test');
load('mnist_background_random_train');
X_train = mnist_background_random_train(:,1:784);
X_test = mnist_background_random_test(:,1:784);
[N_train, D] = size(X_train) % (N_train x D)
[N_test, D] = size(X_test) % (N_test x D)
%%
[coeff, ~, ~, ~, ~, mu] = pca(X_train); % (D x R) = Rows of X_train correspond to observations and columns to variables. 

num_centers = 5; % <-- change
start_centers = 10;
end_centers = 500;
centers = floor(linspace(start_centers, end_centers, num_centers));

train_errors = zeros(1,num_centers);
test_errors = zeros(1,num_centers);
for K=centers
    %% preparing model
    U = coeff(:,1:K); % (D x K) = K pca's of dimension D
    X_train_T = X_train'; % (D x N) = (N_train x D)'
    X_test_T = X_test'; % (D x N) = (N_test x D)'
    X_tilde_train = (U * U' * X_train_T); % (D x N_train)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
    X_tilde_test = (U * U' * X_test_T); % (D x N_test)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
    train_error_PCA = (1/N_train)*norm( X_tilde_train - X_train_T ,'fro')^2;
    test_error_PCA = (1/N_test)*norm( X_tilde_test - X_test_T ,'fro')^2;
end
fig = figure;
plot(num_centers, train_errors);