function [weights, epochErrors] = perceptronConvergenceAlgorithmSteps(input, output, weights, learningRate, epoch)
    
[row, col] = size(input);

lastError = errorPCA(weights,input,output);
epochErrors = [lastError];

for i=1:(epoch*row) 
    idx = mod(i,row) + 1; %make sure that the one index works
    
    y = sign( weights * input(idx,:).');
    weights = weights + learningRate*(output(idx) - y)*input(idx,:);

   %add error per epoch point
   newError = errorPCA(weights, input, output);
   epochErrors = [epochErrors, errorPCA(weights, input, output)];


end
