degree = 30; %set rotation degree
[a,b,class1]=generateData(500,2,0,0,1,0.1, degree); %generate class 1 data
[c,d,class2]=generateData(500,2,0,0,-1,0.1, degree); %generate class 2 data

w0=ones(1,500); %generate bias term
A=[w0;a;b;class1]; %add bias term to class 1
B=[w0;c;d;class2]; %add bias term to class 2
M=[A B]; %merge data to one frame
M=M.'; %transpose the frame
input=M(:,1:3); %input for the PCA 
output=M(:,4); %output for the PCA 

%get weights from the PCA 
[weights,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], 0.05, 30);

%calculate line coeficients from the weights
alpha=-weights(1)/weights(3);
beta=-weights(2)/weights(3);

%plot the results
feedbackPlot(M(:,2),M(:,3),M(:,4),alpha,beta);

data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

bias = ones(row, 1); %generate bias term
input = [bias, data(:,1:2)];
output = data(:,3);

[weights,errors2]=perceptronConvergenceAlgorithm(input, output, [0 0 0], 0.05, 1000);

%calculate line coeficients from the weights
alpha=-weights(1)/weights(3);
beta=-weights(2)/weights(3);

%plot the results
feedbackPlot(data(:,1),data(:,2),data(:,3),alpha,beta);


