files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};

for k = 1:length(files)
    data = importdata(files{k}, '-mat');
    
    %plot the data
    %figure;
    %scatter(data.('x'), data.('t'));

    hidden_layers = [2,3,4,5,7,10,15,25];
    
    errors_test=[];
    errors_train=[];

    for i = 1:length(hidden_layers) 
        hidden_layer = hidden_layers(i);
        

        %5-fold cross validation
        fold = 5;
        indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
        

        for j = 1:fold
            testIdx = (indices == j); 
            trainIdx = ~testIdx;

             %design network, one input layer, one output layer various hidden layers
            [train, error, net] = NeuralNetworkLine(data.('x')(trainIdx), data.('t')(trainIdx), hidden_layer, 4000);

            y = mlpfwd(train, data.('x')(testIdx));
            y_train=mlpfwd(train, data.('x')(trainIdx));
            
            error_test=(rms(data.('t')(testIdx)-y)); 
            errors_test=[errors_test error_test];
            
            error_train=(rms(data.('t')(trainIdx)-y_train)); 
            errors_train=[errors_train error_train];

        end
        
        

        %figure;
        %scatter(data.('x'), data.('t')); hold on;
        %scatter(data.('x')(testIdx), y, 'x');
    end
    
    mean_errors_test=[];
    mean_errors_train=[];
    for i = 1:length(hidden_layers);
        mean_error_test=mean(errors_test((i-1)*fold+1:i*fold));
        mean_errors_test=[mean_errors_test mean_error_test];
        mean_error_train=mean(errors_train((i-1)*fold+1:i*fold));
        mean_errors_train=[mean_errors_train mean_error_train];
    end
    
    figure;
    plot(hidden_layers,mean_errors_test); hold on;
    plot(hidden_layers,mean_errors_train);
    xlabel('number of hidden layers')
    ylabel('average error')
    legend('error test set','error training set')
        
    
end










