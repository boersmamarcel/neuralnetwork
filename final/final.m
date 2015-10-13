data = importdata('dataset_final_assignment.mat');
dataEnriched = data;
%% enrich data
additional = 0;
for t = 1:size(data,1)
    %enrich each time-step
    for a = 1:(size(data,2)/2)
        
       %calculate movement vector
       if t > 1
           dx = data(t, 2*a -1) - data(t-1, 2*a -1) + normrnd(0,1); 
           dy = data(t, 2*a) - data(t-1, 2*a) + normrnd(0,1);
       end
        
       %generate 10 neighbors
       for k = 1:additional
           if t == 1
               %randomly initialize new people with gaussian noise
               x = data(t, 2*a -1) + normrnd(0,10);
               y = data(t, 2*a) + normrnd(0,10);
               dataEnriched(t, size(data,2) + ((a-1)*additional*2)+ (k*2 - 1)) = x;
               dataEnriched(t, size(data,2) + ((a-1)*additional*2) + (k*2)) = y;
           else
               %move with the same direction vector as the original person
               x = dataEnriched(t-1, size(data,2) + ((a-1)*additional*2)+ (k*2 - 1)) + dx;
               y = dataEnriched(t-1, size(data,2) + ((a-1)*additional*2) + (k*2)) + dy;
               dataEnriched(t, size(data,2) + ((a-1)*additional*2)+ (k*2 - 1)) = x;
               dataEnriched(t, size(data,2) + ((a-1)*additional*2) + (k*2)) = y;
           end
            
       end
    end
end

save 'enriched.mat' dataEnriched

