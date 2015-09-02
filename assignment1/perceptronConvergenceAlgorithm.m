function [weights, epochErrors] = perceptronConvergenceAlgorithm(input, output, weights, learningRate, steps)
    
[row, col] = size(input);

epochErrors = [errorPCA(weights,input,output)];

for i=1:steps 
    idx = mod(i,row) + 1;
    
    y = sign( weights * input(idx,:).');
    weights = weights + learningRate*(output(idx) - y)*input(idx,:);

    epochErrors = [epochErrors, errorPCA(weights, input, output)];
end
