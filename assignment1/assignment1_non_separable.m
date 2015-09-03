data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

bias = ones(row, 1); %generate bias term
input = [bias, data(:,1:2)];
output = data(:,3);

epochs = 10;
rate = 0.1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha1=-weights(1)/weights(3);
beta1=-weights(2)/weights(3);


epochs = 10;
rate = 0.5;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights2,errors2]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha2=-weights(1)/weights(3);
beta2=-weights(2)/weights(3);


epochs = 10;
rate = 1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights3,errors3]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha3=-weights(1)/weights(3);
beta3=-weights(2)/weights(3);


%plotting data with separator line
yline1=alpha1+beta1*M(:,2);
yline2=alpha2+beta2*M(:,2);
yline3=alpha3+beta3*M(:,2);

figure
gscatter(data(:,1),data(:,2),data(:,3)); hold on;
plot(M(:,2),yline1); hold on;
plot(M(:,2),yline2); hold on;
plot(M(:,2),yline3); hold on;
   
figure;
%plot errors
plot(errors1); hold on;
plot(errors2); hold on;
plot(errors3); hold on;
xlim([0 epochs]);


