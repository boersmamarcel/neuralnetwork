data = importdata('pics.mat', '-mat');

%construct rbf network
input_layers = 56*46;
output_layers = 1;
hidden_layers = 300;

classes = data.classGlass;

net = rbf(input_layers, hidden_layers, output_layers, 'gaussian');
options = zeros(1,14);
net2 = rbfsetbf(net, options, data.pics);

net3 = rbftrain(net2, options, data.pics, data.classGlass.');

y = rbffwd(net3, data.pics);
classEst = y>0;

error = rms(classEst - data.classGlass.');
disp(error);