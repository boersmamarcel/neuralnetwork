%use GLM for the separable dataset
degree = 30; %set rotation degree
[a,b,class1]=generateData(500,1,0.0,0.5,1,0.1, degree); %generate class 1 data
[c,d,class2]=generateData(500,1,0.0,-0.5,-1,0.1, degree); %generate class 2 data

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

epochs = 10;
rate = 1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);
