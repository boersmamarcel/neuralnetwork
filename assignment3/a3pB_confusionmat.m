data = importdata('pics.mat', '-mat');
net=mlp(56*46, 1500, 1, 'linear');
finalNet=mlptrain(net, data.pics, data.classGlass.', 100)
y = mlpfwd(finalNet, data.pics);
y=y>0
M=[y,data.classGlass.'];
confusionmat(M(:,1),M(:,2))