%apply SVM to our own generated data

data = importdata('owndata.mat', '-mat');

[yp,alpha,b,gam,sig2,model] = lssvm([data(:,2),data(:,3)],data(:,4),'c');

