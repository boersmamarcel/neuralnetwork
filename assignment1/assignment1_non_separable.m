data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

bias = ones(row, 1); %generate bias term
input = [bias, data(:,1:2)];
output = data(:,3)*2 - 1; %transform 0->-1 and 1->1

epochs = 10;
rate = 0.1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha1=-weights1(1)/weights1(3);
beta1=-weights1(2)/weights1(3);




%plotting data with separator line
yline1=alpha1+beta1*data(:,1);

figure
gscatter(data(:,1),data(:,2),data(:,3)); hold on;
plot(data(:,1),yline1); hold on;
   
figure;
%plot errors
plot(errors1); hold on;
xlim([0 epochs]);


