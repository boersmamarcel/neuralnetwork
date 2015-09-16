files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};

for k = 1:length(files)
    data = importdata(files{k}, '-mat');
    
    %plot the data
    %figure;
    %scatter(data.('x'), data.('t'));

    hidden_layers = [2,3,4,5,7,10,15,25];
    
    errors=[];

    for i = 1:length(hidden_layers) 
        hidden_layer = hidden_layers(i);
        

        %5-fold cross validation
        fold = 5;
        indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
        

        for j = 1:fold
            testIdx = (indices == j); 
            trainIdx = ~testIdx;

             %design network, one input layer, one output layer various hidden layers
            [train, error, net] = NeuralNetworkLine(data.('x')(trainIdx), data.('t')(trainIdx), hidden_layer, 1500);

            y = mlpfwd(train, data.('x')(testIdx));
            
            error=(rms(data.('t')(testIdx)-y)); 
            errors=[errors error];

        end
        
        

        figure;
        scatter(data.('x'), data.('t')); hold on;
        scatter(data.('x')(testIdx), y, 'x');
    end
    
    mean_errors=[];
    for i = 1:length(hidden_layers);
        mean_error=mean(errors((i-1)*5+1:i*5));
        mean_errors=[mean_errors mean_error];
    end
    
    figure;
    plot(hidden_layers,mean_errors);
    xlabel('number of hidden layers')
    ylabel('average error')
        
    
end









