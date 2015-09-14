function [error] = errorPCA(weights, input, output)
    [row, col] = size(input);
    
    error = 0;
    %calculate error for each input term
    for i=1:row 
        y = sign( weights * input(i,:).');
        error = error + (output(i) - y)^2;
    end
    
    %average error
    error = error/row;