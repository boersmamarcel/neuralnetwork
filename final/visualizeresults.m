% Spatial crowd behaviour simulator using a neural network

% Mark Hoogendoorn, edited by Marcel Boersma (October 2015)

% May 2012



%clear all;


%%train network with optimal parameters
% 4500 hidden nodes, 100 training cycles default learning rate

dataNN = importdata('featuredata_non_timeseries.mat');

input = dataNN(:, [2 3 4 5 8 9]);
output = dataNN(:, 6:7);

input_layers = size(input,2);
output_layers = size(output,2);


net = mlp(input_layers, hidden_layer, output_layers, 'linear');


options = zeros(1,20);
options(1) = -1; %suppress warnings

net = netopt(net, options, input, output, 'scg');%scaled conjugate gradient/standard gradient


cycles = 100;

[trainNet, error] = mlptrain(net, input, output, cycles);

predictedNN = mlpfwd(trainNet, input);

load 'dataset_final_assignment.mat';
% data = importdata('enriched.mat');

% grid size

max_x = 600;

max_y = 800;

end_time = size(data,1);

number_of_agents = size(data,2)/2;

% pre-process data (mirror all people over y-axis, !only if not already done!)

if data(1,2) > max_y/2

  for i = 1:end_time

    for a = 1:size(data(1,:),2)/2

      data(i,2*a) = max_y-data(i,2*a);

    end

  end

end



% relevant points (e.g. corners of buildings)

% NOTE: "max_y" operation was added, because image needs to be y-mirrorred

points = [546.8, max_y-478.0;

          507.5, max_y-330.6;

          240.6, max_y-218.6;

          184.7, max_y-331.3];



% sources of panic (e.g. shouting individual)

source = [542.0, max_y-439.0];

      

% solid lines that cannot be passed (buildings, fences, ...)

% NOTE: some lines have been made longer than represented in the data

lines = [321.2, max_y-314.5, 240.0, 300.0;          % was 321.2, max_y-314.5, 286.1, max_y-396.9;

         321.2, max_y-314.5, 275.5, max_y-292.0;

         383.0, max_y-336.4, 342.0, 300.0;          % was 383.0, max_y-336.4, 365.2, max_y-407.2

         383.0, max_y-336.4, 600.0, 358.0;          % was 383.0, max_y-336.4, 431.2, max_y-359.6

         385.3, max_y-321.6, 448.0, max_y-347.4;

         448.0, max_y-347.4, 449.0, max_y-313.9;

         449.0, max_y-313.9, 390.5, max_y-292.0;

         390.5, max_y-292.0, 385.3, max_y-321.6];



%%%%%%%%%%%%%%%%%%%%%%

%%% Pre-processing %%%

%%%%%%%%%%%%%%%%%%%%%%


% calculate slopes and bases of solid lines

for l = 1:size(lines, 1)

    slopes(l) = (lines(l, 4) - lines(l, 2)) / (lines(l, 3) - lines(l, 1));

    bases(l) = lines(l, 2) - slopes(l) * lines(l, 1);

end



% place people in the environment



% Do some drawing....
clf;
hold on;
grid on;
set(gcf, 'Position', [265 5 750 1000])
set(gca,'DataAspectRatio',[1 1 1]);
axis([0 max_x+1 0 max_y+1]);
    
% draw lines

for l = 1:size(lines, 1)
  line([lines(l,1) lines(l,3)],[lines(l,2) lines(l,4)],'Color',[0 0 0],'LineStyle','-');
end

% draw environmental objects (circle)

radius = sqrt((386.9-374.3)^2+(208.9-257.2)^2);
t=(0:50)*2*pi/50;
x=radius*cos(t)+386.9;
y=radius*sin(t)+max_y-208.9;
plot(x,y,'Color',[0 0 0]);

% draw relevant points

for l = 1:size(points, 1)
    plot(points(l,1),points(l,2),'Color',[0 0 0],'Marker','.','MarkerSize',20);
end

% draw source

plot(source(1),source(2),'Color',[1 0 0],'Marker','.','MarkerSize',20);

% draw agents 1,2,3

agent1x = [];
agent1y = [];
agent2x = [];
agent2y = [];
agent3x = [];
agent3y = [];

for t = 1:end_time
    agent1x = [agent1x data(t,(2*1)-1)];
    agent1y = [agent1y data(t,(2*1))];
    
    agent2x = [agent2x data(t,(2*25)-1)];
    agent2y = [agent2y data(t,(2*25))];
    
    agent3x = [agent3x data(t,(2*34)-1)];
    agent3y = [agent3y data(t,(2*34))];
end

agent1xNN = [];
agent1yNN = [];
agent2xNN = [];
agent2yNN = [];
agent3xNN = [];
agent3yNN = [];

for t = 0:(end_time-1)
    idx = t*number_of_agents; 
    
    agent1xNN = [agent1xNN dataNN(idx+1, 2) + predictedNN(idx+1,1)];
    agent1yNN = [agent1yNN max_y-dataNN(idx+1, 3) + predictedNN(idx+1,2)];
    
    agent2xNN = [agent2xNN dataNN(idx+25, 2) + predictedNN(idx+25,1)];
    agent2yNN = [agent2yNN max_y-dataNN(idx+25, 3) + predictedNN(idx+25,2)];
    
    agent3xNN = [agent3xNN dataNN(idx+34, 2) + predictedNN(idx+34,1)];
    agent3yNN = [agent3yNN max_y-dataNN(idx+34, 3) + predictedNN(idx+34,2)];
    
end


plot(agent1x, agent1y, '-r', agent2x, agent2y, '-r', agent3x, agent3y, '-r');hold on;
plot(agent1xNN, agent1yNN, '-b', agent2xNN, agent2yNN, '-b', agent3xNN, agent3yNN, '-b'); 

saveas(gcf,strcat('paths','.png'));
