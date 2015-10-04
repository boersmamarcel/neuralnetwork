data = importdata('dataset_final_assignment.mat');
dataEnriched = data;
%% enrich data
additional = 100;
for t = 1:size(data,1)
    %enrich each time-step
    for a = 1:(size(data,2)/2)
        
       %generate 10 neighbors
       for k = 1:additional
            x = data(t, 2*a -1) + normrnd(0,3);
            y = data(t, 2*a) + normrnd(0,3);
            dataEnriched(t, size(data,2) + ((a-1)*additional*2)+ (k*2 - 1)) = x;
            dataEnriched(t, size(data,2) + ((a-1)*additional*2) + (k*2)) = y;
            
       end
    end
end

save 'enriched.mat' dataEnriched

