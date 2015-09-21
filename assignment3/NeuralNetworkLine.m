function [trainNet, error, net] = NeuralNetworkLine( input, output, hidden_layers, cycles, gradtype)
    input_layers = 1;
    output_layers = 1;
    net = mlp(input_layers, hidden_layers, output_layers, 'linear');

    options = zeros(1,20);
    options(1) = -1; %suppress warnings
    net = netopt(net, options, input, output, gradtype);%scaled conjugate gradient/standard gradient
    
    [trainNet, error] = mlptrain(net, input, output, cycles);
end

