data = importdata('enriched.mat');

numberOfFeatures = 4;
agents = size(data,2)/2;
featureMatrix = zeros(size(data,1)*agents, 4); %time step * agents = matrix

%% Feature 1: Distance to source
max_y = 800;
source = [542.0, max_y-439.0];

for t = 1:size(data,1)
    %enrich each time-step
    for a = 1:(size(data,2)/2)
        x = data(t, 2*a -1);
        y = data(t, 2*a);
        
        distanceToSource = sqrt((x - source(1))^2 + (y-source(2))^2);
        %distance to source at time t for agent a
        featureMatrix((t-1)*agents + a,1) = distanceToSource;
        
    end
end


%% Feature 2: Distance to closest fence?



%% Feature 3: Time since shout?


%% Feature 4: Distance to monument
monument = [386.9 208.9];
for t = 1:size(data,1)
    %enrich each time-step
    for a = 1:(size(data,2)/2)
        x = data(t, 2*a -1);
        y = data(t, 2*a);
        
        distanceToMonument = sqrt((x - monument(1))^2 + (y-monument(2))^2);
        %distance to source at time t for agent a
        featureMatrix((t-1)*agents + a, 4) = distanceToMonument;
        
    end
end
