function [trainNet, error, net] = NeuralNetworkLine( input, output, hidden_layers, cycles )
    input_layers = 1;
    output_layers = 1;
    net = mlp(input_layers, hidden_layers, output_layers, 'linear');

    [trainNet, error] = mlptrain(net, input, output, cycles);
end

