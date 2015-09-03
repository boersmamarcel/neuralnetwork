degree = 15; %set rotation degree
[a,b,class1]=generateData(500,1,0.0,0.5,1,0.1, degree); %generate class 1 data
[c,d,class2]=generateData(500,1,0.0,-0.5,-1,0.1, degree); %generate class 2 data

w0=ones(1,500); %generate bias term
A=[w0;a;b;class1]; %add bias term to class 1
B=[w0;c;d;class2]; %add bias term to class 2
M=[A B]; %merge data to one frame
M=M.'; %transpose the frame
input=M(:,1:3); %input for the PCA 
output=M(:,4); %output for the PCA 

epochs = 10;
rate = 0.01;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights1,errors1]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha1=-weights1(1)/weights1(3);
beta1=-weights1(2)/weights1(3);


epochs = 10;
rate = 0.05;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights2,errors2]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha2=-weights2(1)/weights2(3);
beta2=-weights2(2)/weights2(3);


epochs = 10;
rate = 0.1;

fprintf('Learning rate: %f and epoch: %d \n', rate, epochs);

%get weights from the PCA 
[weights3,errors3]=perceptronConvergenceAlgorithm(input, output, [0 0 0], rate, epochs);

%calculate line coeficients from the weights
alpha3=-weights3(1)/weights3(3);
beta3=-weights3(2)/weights3(3);


%plotting data with separator line
yline1=alpha1+beta1*M(:,2);
yline2=alpha2+beta2*M(:,2);
yline3=alpha3+beta3*M(:,2);

figure
gscatter(M(:,2),M(:,3),M(:,4)); hold on;
plot(M(:,2),yline1); hold on;
plot(M(:,2),yline2); hold on;
plot(M(:,2),yline3); hold on;
   
figure;
%plot errors
plot(errors1); hold on;
plot(errors2); hold on;
plot(errors3); hold on;
xlim([0 epochs]);
legend('0.1', '0.5', '1');





