function [ error ] = distance( imgs1, imgs2 )
    [row1, col1] = size(imgs1);
    [row2, col2] = size(imgs2);
     
    error = 0;
    if row1 == row2 && col1==col2 
        for i = 1:row1
            error = error + sqrt((1/col1)*sum((imgs1(i,:)-imgs2(i,:)).^2));
        end
        
        error = (1/row1)*error;
    else
        disp('Invalid dimensions of matrices');
        error = -1;
    end
end

