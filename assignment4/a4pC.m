data = importdata('pics.mat', '-mat');


[yp,alpha,b,gam,sig2,model] = lssvm(data.pics, data.classGlass.','c');

