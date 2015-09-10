function [weights, epochErrors] = perceptronConvergenceAlgorithm(input, output, weights, learningRate, epoch)
    
[row, col] = size(input);

lastError = errorPCA(weights,input,output);
epochErrors = [lastError];

for i=1:(epoch*row) 
    idx = mod(i,row) + 1; %make sure that the one index works
    
    y = sign( weights * input(idx,:).');
    weights = weights + learningRate*(output(idx) - y)*input(idx,:);

    if mod(i,row) == 0
       %add error per epoch point
       newError = errorPCA(weights, input, output);
       epochErrors = [epochErrors, errorPCA(weights, input, output)];

       if lastError == newError
           %last error is equal to the new error for this epoch, then the
           %algorithm has converged. Thus, break the loop
          break; 
       end
       
       %update last known error
       lastError = newError;
    end
end
