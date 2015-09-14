%use GLM for the non separable dataset
data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

input = data(:,1:2);
output = data(:,3)*2 - 1;


net = glm(2, 1, 'linear');
options = zeros(15,1);
train = glmtrain(net, options, input, output);


weights = [train.('b1'), train.('w1').'];
input = [ones(length(input), 1), input];

error = errorPCA(weights, input, output);

%% USE Perceptron convergence algorithm

epochs = 10;
rate = 1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%% Compare errors PCA and GLM
figure
bar([error, errors1(length(errors1))])
set(gca,'XTickLabel',{'Error GLM', 'Error PCA'})
legend('RMSE score');


%% Show the decision boundary that has been found, and make a comparison between the results you have obtained using the LMS approach and the perceptron.

%PCA line
alpha1=-weights1(1)/weights1(3);
beta1=-weights1(2)/weights1(3);

%GLM line
alpha=-weights(1)/weights(3);
beta=-weights(2)/weights(3);

%plotting data with separator line
yline1=alpha1+beta1*data(:,1);
yline2=alpha+beta*data(:,1);

figure
plot(data(:,1),yline1); hold on;
plot(data(:,1),yline2); hold on;
gscatter(data(:,1),data(:,2),data(:,3)); hold on;
legend('PCA boundary', 'GLM boundary');


