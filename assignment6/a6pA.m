X = exprnd(2,[1,500]);
Y = randn(500,1);
scatter(16-X,Y);
M=[X.' Y];

%set parameters
epochs=100; 
initial_learning_rate=0.9;
final_learning_rate=0.05;

net=som(500,[10,10]);

options(14)=epochs ;
options(18) = initial_learning_rate;  
options(16) = final_learning_rate; 

net2=somtrain(net,options,M.');

options(14) = 400;
options(18) = 0.05;
options(16) = 0.01;
options(17) = 0;
options(15) = 0;
net3 = somtrain(net2, options, M.');