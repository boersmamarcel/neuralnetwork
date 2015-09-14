data = importdata('line.mat', '-mat');

%plot the data
figure;
scatter(data.('x'), data.('t'));

hidden_layers = [2,3,4,5,7,10,15,25];

for i = 1:length(hidden_layers) 
    hidden_layer = hidden_layers(i);
    
    %design network, one input layer, one output layer various hidden layers
    [train, error, net] = NeuralNetworkLine(data.('x'), data.('t'), i, 10000);
    
    y = mlpfwd(net, data.('x'));    
    
    figure;
    scatter(data.('x'), data.('t')); hold on;
    scatter(data.('x'), y, 'x');
    
end

%5-fold cross validation
Indices = crossvalind('Kfold', length(data.('x')), 5); %randomly assigns indices
