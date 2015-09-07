%use GLM for the non separable dataset
data = importdata('two_class_example_not_separable.dat');

[row,col] = size(data);

input = data(:,1:2);
output = data(:,3)*2 - 1;


net = glm(2, 1, 'linear');
options = zeros(15,1);
train = glmtrain(net, options, input, output);


error = glmerr(net, input, output);

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