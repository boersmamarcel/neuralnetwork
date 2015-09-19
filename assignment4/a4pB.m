%apply SVM to our own generated data

data = importdata('owndata.mat', '-mat');


X = data(:,2:3);
Y = data(:, 4);

type='c'; %classification
kernel = 'RBF_kernel'; %RBF_kernel/lin_kernel/poly_kernel
dataprocessing = 'preprocess'; % preprocess/original
%GAM: regularization parameter
% for gam low minimizing of the
% complexity of the model is emphasized, for gam high, good fitting
% of the training data points is stressed.
gam = 1;

if strcmp(kernel, 'poly_kernel')
    degree = 10;
    highlowbalance = 0.5;
    model = {X,Y,type,gam,[highlowbalance degree],kernel};
   
elseif strcmp(kernel, 'RBF_kernel')
    sig2 = 0.2;
    model = {X,Y,type,gam,sig2,kernel,dataprocessing};
        
else
    model = {X,Y,type,gam,[],kernel,dataprocessing};
    
end

[alpha,b] = trainlssvm(model);
Ytest = simlssvm(model, {alpha, b}, X);
figure; plotlssvm(model, {alpha, b});

