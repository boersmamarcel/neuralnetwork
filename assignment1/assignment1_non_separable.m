data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

bias = ones(row, 1); %generate bias term
input = [bias, data(:,1:2)];
output = data(:,3);

[weights,errors2]=perceptronConvergenceAlgorithm(input, output, [0 0 0], 0.1, 100);

%calculate line coeficients from the weights
alpha=-weights(1)/weights(3);
beta=-weights(2)/weights(3);

%plot the results
feedbackPlot(data(:,1),data(:,2),data(:,3),alpha,beta);

figure
%plot errors
plot(errors2);

