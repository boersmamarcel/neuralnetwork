%apply SVM to our own generated data

data = importdata('owndata.mat', '-mat');

% [yp,alpha,b,gam,sig2,model] = lssvm([data(:,2),data(:,3)],data(:,4),'c', 'poly_kernel');

X = data(:,2:3);
Y = data(:, 4);
gam = 10;
sig2 = 0.2;
type='c'; %classification
kernel = 'RBF_kernel'; %RBF_kernel/lin_kernel/poly_kernel
dataprocessing = 'preprocess'; % preprocess/original
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,kernel,dataprocessing});

Ytest = simlssvm({X,Y,type,gam,sig2,kernel,dataprocessing},{alpha,b},X);
figure; plotlssvm({X,Y,type,gam,sig2,kernel,dataprocessing},{alpha,b});