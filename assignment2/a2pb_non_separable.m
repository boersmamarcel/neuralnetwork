%use GLM for the non separable dataset
data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

input = data(:,1:2);
output = data(:,3)*2 - 1;


net = glm(2, 1, 'linear');
options = zeros(15,1);
train = glmtrain(net, options, input, output);


errorglm = glmerr(net, input, output);

weights = [train.('b1'), train.('w1').'];
input = [ones(length(input), 1), input];

error = errorPCA(weights, input, output);

%% USE Perceptron convergence algorithm

epochs = 10;
rate = 1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);