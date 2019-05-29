%% Initialize
close all
clc


clear
d1 = csvread('mnist_train.csv', 1, 0);                   % read train.csv
d2 = csvread('mnist_test.csv', 1, 0);                   % read test.csv

tr = [d1; d2];

testFract = 7;                                  % 1/testFract of tr is for test set
%% Sample plot

figure                                          % plot images
colormap(gray)                                  % set to grayscale
for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
    digit = reshape(tr(i, 2:end), [28,28])';    % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(tr(i, 1)))                    % show the label
end

%% Data preparation

n = size(tr, 1);                    % number of samples in the dataset
targets  = tr(:,1);                 % 1st column is |label|
targets(targets == 0) = 10;         % use '10' to present '0'
targetsd = dummyvar(targets);       % convert label into a dummy variable
inputs = tr(:,2:end);               % the rest of columns are predictors

inputs = inputs';                   % transpose input
targets = targets';                 % transpose target
targetsd = targetsd';               % transpose dummy variable

rng(1);                             % for reproducibility
c = cvpartition(n,'Holdout',floor(n/testFract));   % hold out 1/3 of the dataset

Xtrain = inputs(:, training(c));    % 2/3 of the input for training
Ytrain = targetsd(:, training(c));  % 2/3 of the target for training
Xtest = inputs(:, test(c));         % 1/3 of the input for testing
Ytest = targets(test(c));           % 1/3 of the target for testing
Ytestd = targetsd(:, test(c));      % 1/3 of the dummy variable for testing

%% Train - 1 Hidden Layer
x = Xtrain;                                              % inputs
t = Ytrain;                                              % targets

hiddenLayerSize = [256];                                 % number of hidden layer neurons
% trainFcn = 'traingd';                                    % gradient descent backpropagation
trainFcn = 'trainscg';                                   % scaled conjugate gradient
performFcn = 'crossentropy';                             % Loss: crossentropy

net = patternnet(hiddenLayerSize,trainFcn,performFcn);   % pattern recognition network
net.performParam.regularization = 0;                     % 0<1: weight regularization parameter
net.performParam.normalization = 'none';                 % no error regularization

% Early stopping:
net.divideFcn = 'dividerand';                          % randomly divide data in three subsets
net.trainParam.max_fail = 6;                    % # of itr for early stopping (valid. err. increase)
net.divideParam.trainRatio = 70/100;                 % 70% of data for training
net.divideParam.valRatio = 15/100;                   % 15% of data for validation
net.divideParam.testRatio = 15/100;                  % 15% of data for testing
% see Early Stopping in 
% https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html

% net.divideFcn = 'dividetrain';                         % divide data in one subset (training only)

tic
[net, tr]= train(net, x, t);                             % train the network
toc
model = net;                                             % store the trained network
p = net(Xtest);                                          % predictions
[~, p] = max(p);                                         % predicted labels
disp('score for model 1:')
score = sum(Ytest == p) / length(Ytest)                  % categorization accuracy


%% Train - 2 Hidden Layer
x = Xtrain;                                              % inputs
t = Ytrain;                                              % targets

hiddenLayerSize = [256, 64];                             % number of hidden layer neurons
trainFcn = 'traingd';                                    % gradient descent backpropagation
% trainFcn = 'trainscg';                                   % scaled conjugate gradient
performFcn = 'crossentropy';                             % Loss: crossentropy

net2 = patternnet(hiddenLayerSize,trainFcn,performFcn);   % pattern recognition network
net2.performParam.regularization = 0;                     % 0<1: weight regularization parameter
net2.performParam.normalization = 'none';                 % no error regularization

% Early stopping:
net2.divideFcn = 'dividerand';                          % randomly divide data in three subsets
net2.trainParam.max_fail = 6;                    % # of itr for early stopping (valid. err. increase)
net2.divideParam.trainRatio = 70/100;                 % 70% of data for training
net2.divideParam.valRatio = 15/100;                   % 15% of data for validation
net2.divideParam.testRatio = 15/100;                  % 15% of data for testing
% see Early Stopping in 
% https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html

% net2.divideFcn = 'dividetrain';                         % divide data in one subset (training only)

tic
[net2, tr2]= train(net2, x, t);                             % train the network
toc
model2 = net2;                                             % store the trained network
p2 = net2(Xtest);                                          % predictions
[~, p2] = max(p2);                                         % predicted labels
disp('score for model 2:')
score2 = sum(Ytest == p2) / length(Ytest)                  % categorization accuracy


%% Train - 3 Hidden Layer
x = Xtrain;                                              % inputs
t = Ytrain;                                              % targets

hiddenLayerSize = [256, 64, 16];                         % number of hidden layer neurons
trainFcn = 'traingd';                                    % gradient descent backpropagation
% trainFcn = 'trainscg';                                   % scaled conjugate gradient
performFcn = 'crossentropy';                             % Loss: crossentropy

net3 = patternnet(hiddenLayerSize,trainFcn,performFcn);   % pattern recognition network
net3.performParam.regularization = 0;                     % 0<1: weight regularization parameter
net3.performParam.normalization = 'none';                 % no error regularization

% Early stopping:
net3.divideFcn = 'dividerand';                          % randomly divide data in three subsets
net3.trainParam.max_fail = 6;                    % # of itr for early stopping (valid. err. increase)
net3.divideParam.trainRatio = 70/100;                 % 70% of data for training
net3.divideParam.valRatio = 15/100;                   % 15% of data for validation
net3.divideParam.testRatio = 15/100;                  % 15% of data for testing
% see Early Stopping in 
% https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html

% net3.divideFcn = 'dividetrain';                         % divide data in one subset (training only)

tic
[net3, tr3]= train(net3, x, t);                             % train the network
toc
model3 = net3;                                             % store the trained network
p3 = net3(Xtest);                                           % predictions
[~, p3] = max(p3);                                         % predicted labels
disp('score for model 1:')
score3 = sum(Ytest == p3) / length(Ytest)                  % categorization accuracy


