%use GLM for the separable dataset
degree = 0; %set rotation degree
[a,b,class1]=generateData(1500,1,-0.5,0.05,1,0.1, degree); %generate class 1 data
[c,d,class2]=generateData(1500,1,0.5,-0.05,-1,0.1, degree); %generate class 2 data

A=[a;b;class1]; %add bias term to class 1
B=[c;d;class2]; %add bias term to class 2
M=[A B]; %merge data to one frame
M=M.'; %transpose the frame
input=M(:,1:2); %input for the PCA 
output=M(:,3); %output for the PCA

net = glm(2, 1, 'linear');
options = zeros(15,1);
train = glmtrain(net, options, input, output);


weights = [train.('b1'), train.('w1').'];
input = [ones(length(input), 1), input];

error = errorPCA(weights, input, output);

%% USE Perceptron convergence algorithm

epochs = 1000;
rate = 1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%% Compare errors PCA and GLM
figure
bar([error, errors1(length(errors1))])
set(gca,'XTickLabel',{'Error GLM', 'Error PCA'})
legend('RMSE score');
ylim([0,1]);

%% Show the decision boundary that has been found, and make a comparison between the results you have obtained using the LMS approach and the perceptron.

%PCA line
alpha1=-weights1(1)/weights1(3);
beta1=-weights1(2)/weights1(3);

%GLM line
alpha=-weights(1)/weights(3);
beta=-weights(2)/weights(3);

%plotting data with separator line
yline1=alpha1+beta1*M(:,2);
yline2=alpha+beta*M(:,2);

figure
plot(M(:,2),yline1); hold on;
plot(M(:,2),yline2); hold on;
gscatter(M(:,1),M(:,2),M(:,3)); 
legend('PCA boundary', 'GLM boundary');