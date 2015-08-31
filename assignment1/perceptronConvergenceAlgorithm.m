function [weights] = perceptronConvergenceAlgorithm(input, output, weights, learningRate, steps)
    
[row, col] = size(input);

for i=1:steps 
    idx = mod(i,row) + 1;
    
    y = sign( weights * input(idx,:).');
    weights = weights + learningRate*(output(idx) - y)*input(idx,:);

end
