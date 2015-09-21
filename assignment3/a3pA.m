files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};
gradients = {'graddesc', 'scg', 'quasinew'};

for k = 1:length(files)
    
    mean_errors_for_gradient_train = {};
    mean_errors_for_gradient_test  = {};
    
    for g = 1:length(gradients)
        data = importdata(files{k}, '-mat');

        
        hidden_layers = [2,3,4,5,7,10,15,25];


        errors_test=[];
        errors_train=[];
        
        avg_mean_errors_train = [];
        avg_mean_errors_test = [];

        for i = 1:length(hidden_layers) 
            hidden_layer = hidden_layers(i);

            mean_errors_test = [];
            mean_errors_train = [];
            
            
            for l = 1:10

                %5-fold cross validation
                fold = 5;
                indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices


                for j = 1:fold
                    testIdx = (indices == j); 
                    trainIdx = ~testIdx;

                     %design network, one input layer, one output layer various hidden layers
                    [train, error, net] = NeuralNetworkLine(data.('x')(trainIdx), data.('t')(trainIdx), hidden_layer, 100, gradients{g});

                    y = mlpfwd(train, data.('x')(testIdx));
                    y_train=mlpfwd(train, data.('x')(trainIdx));

                    error_test=(rms(data.('t')(testIdx)-y)); 
                    errors_test=[errors_test error_test];

                    error_train=(rms(data.('t')(trainIdx)-y_train)); 
                    errors_train=[errors_train error_train];

                end
                
                %per hidden neuron
                mean_errors_test = [mean_errors_test mean(errors_test)];
                mean_errors_train = [mean_errors_train mean(errors_train)];
            
            
            end
            %mean over n iterations per neuron
            
            avg_mean_errors_train = [avg_mean_errors_train mean(mean_errors_train)];
            avg_mean_errors_test = [avg_mean_errors_test mean(mean_errors_test)];

        end

        
        %save errors per gradient type
        mean_errors_for_gradient_train{g} = avg_mean_errors_train;
        mean_errors_for_gradient_test{g} = avg_mean_errors_test;
    end
 
    
    %plot scg and graddesc in one image
    legendText = {};
    shapesTrain = {'--o','*', '--'};
    shapesTest = {'-o', '--*', 'x'};
    figure;
    for g=1:length(gradients)
        disp(gradients{g});
        plot(hidden_layers,mean_errors_for_gradient_test{g}, shapesTrain{g}); hold on;
        plot(hidden_layers,mean_errors_for_gradient_train{g}, shapesTest{g}); hold on;
        legendText = [legendText, strcat(gradients{g},'-test'), strcat(gradients{g},'-train')];
    end
    
    xlabel('number of hidden layers')
    ylabel('average error')
    title(files{k});
    ylim([0 0.3]);
    legend(legendText);
    
    saveas(gcf,strcat(files{k},'.png'));
    
end










