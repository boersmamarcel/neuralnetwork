data = importdata('pics.mat', '-mat');

X = data.pics;
Y = data.classGlass.';
gam = 10;
sig2 = 0.2;
type='c'; %classification
kernel = 'RBF_kernel'; %RBF_kernel/lin_kernel/poly_kernel
dataprocessing = 'preprocess'; % preprocess/original
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,kernel,dataprocessing});

Ytest = simlssvm({X,Y,type,gam,sig2,kernel,dataprocessing},{alpha,b},X);
figure; plotlssvm({X,Y,type,gam,sig2,kernel,dataprocessing},{alpha,b});