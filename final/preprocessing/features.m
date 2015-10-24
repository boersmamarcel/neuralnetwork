data = importdata('../data/enriched.mat');

numberOfFeatures = 9;
agents = size(data,2)/2;
featureMatrix = zeros(size(data,1)*agents, numberOfFeatures); %time step * agents = matrix

%% agent
for t = 1:size(data,1)
    for a = 1:(size(data,2)/2)
        x = data(t, 2*a -1);
        y = data(t, 2*a);
        
        featureMatrix((t-1)*agents + a,1) = a;
    end
end



%% x,y not as time-series
for t = 1:size(data,1)
    for a = 1:(size(data,2)/2)
        x = data(t, 2*a -1);
        y = data(t, 2*a);
        
        featureMatrix((t-1)*agents + a,2) = x;
        featureMatrix((t-1)*agents + a,3) = y;
    end
end

%% dx,dy not as time-series
for t = 1:size(data,1)
    for a = 1:(size(data,2)/2)
        if t == 1
           dx = 0;
           dy = 0;
        else
            dx = data(t, 2*a -1)-data(t-1, 2*a -1);
            dy = data(t, 2*a)-data(t-1, 2*a);
        end
        
        
        featureMatrix((t-1)*agents + a,6) = dx;
        featureMatrix((t-1)*agents + a,7) = dy;
    end
end


%% dx(t-1),dy(t-1) not as time-series
for t = 1:size(data,1)
    for a = 1:(size(data,2)/2)
        if t == 1
           dx = 0;
           dy = 0;
        else
            dx = featureMatrix((t-2)*agents + a,6);
            dy = featureMatrix((t-2)*agents + a,7);
        end
        
        
        featureMatrix((t-1)*agents + a,4) = dx;
        featureMatrix((t-1)*agents + a,5) = dy;
    end
end



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
        featureMatrix((t-1)*agents + a,8) = distanceToSource;
        
    end
end


%% Feature 2: Distance to monument
monument = [386.9 208.9];
for t = 1:size(data,1)
    %enrich each time-step
    for a = 1:(size(data,2)/2)
        x = data(t, 2*a -1);
        y = data(t, 2*a);
        
        distanceToMonument = sqrt((x - monument(1))^2 + (y-monument(2))^2);
        %distance to source at time t for agent a
        featureMatrix((t-1)*agents + a, 9) = distanceToMonument;
        
    end
end

save 'featuredata_non_timeseries.mat' featureMatrix
