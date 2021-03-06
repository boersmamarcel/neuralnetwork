data = importdata('sinus.mat', '-mat');

%plot the data
figure;
scatter(data.('x'), data.('t'));

hidden_layers = [2,3,4,5,7,10,15,25];

for i = 1:length(hidden_layers) 
    hidden_layer = hidden_layers(i);
    
    %5-fold cross validation
    fold = 5;
    indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
    
    for j = 1:fold
        testIdx = (indices == j); 
        trainIdx = ~testIdx;
        
         %design network, one input layer, one output layer various hidden layers
        [train, error, net] = NeuralNetworkLine(data.('x')(trainIdx), data.('t')(trainIdx), i, 10000);
        
        y = mlpfwd(net, data.('x')(testIdx));
    
                
        figure;
        scatter(data.('x'), data.('t')); hold on;
        scatter(data.('x')(testIdx), y, 'x');
    end
    
    
end


