data = importdata('featuredata_non_timeseries.mat');

input = data(:, [2 3 4 5 8 9]);
output = data(:, 6:7);


input_layers = size(input,2);
hidden_layers = 200;
output_layers = size(output,2);


net = mlp(input_layers, hidden_layers, output_layers, 'linear');



options = zeros(1,20);
options(1) = -1; %suppress warnings
gradtype='scg';

net = netopt(net, options, input, output, gradtype);%scaled conjugate gradient/standard gradient


cycles = 10000;

[trainNet, error] = mlptrain(net, input, output, cycles);

outputNetwork = mlpfwd(trainNet, input);
dX = outputNetwork(:,1);
dY = outputNetwork(:,2);


error_x = (1/size(data,1))*sum(sqrt((dX - data(:,6)).^2));
error_y = (1/size(data,1))*sum(sqrt((dY - data(:,7)).^2));

