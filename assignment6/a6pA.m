X = exprnd(2,[1,500]);
X=1-(X/max(X));
Y = randn(500,1);
Y=Y/max(Y)
figure;
scatter(X,Y,'bo'); hold on; 

M=[X.' Y];

net=som(2,[50,50]);

options = foptions;
options(14) = 1000  %epochs ;
options(18) = 0.9   %initial_learning_rate;  
options(16) = 0.05  %final_learning_rate; 
options(17) = 8;    %Initial neighbourhood size
options(15) = 1;    %Final neighbourhood size

net2=somtrain(net,options,M);
c2 = sompak(net2);
figure;
plot(c2(:, 1), c2(:, 2), 'r*'); hold on;


options(14) = 400;  %epochs
options(18) = 0.01; %initial_learning_rate; 
options(16) = 0.01; %final_learning_rate;
options(17) = 1;    %Initial neighbourhood size
options(15) = 0;    %Final neighbourhood size
net3 = somtrain(net2, options, M);
c3 = sompak(net3);
plot(c3(:, 1), c3(:, 2), 'g^');