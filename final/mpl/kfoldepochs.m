%% settings : setup experiment
%gradients = {'graddesc', 'scg', 'quasinew'};
gradients = {'scg'};

hidden_nodes = 4500;
% cycles = [10 20 40 80 160 320];
cycles = [1 2 3 4 5 10 20];

data = importdata('../data/featuredata_non_timeseries.mat');
    
mean_errors_for_gradient_train = {};
mean_errors_for_gradient_test  = {};

for g = 1:length(gradients)
    disp('Start training for gradient');
    disp(gradients{g});
    
    input = data(:, [2 3 4 5 8 9]);
    output = data(:, 6:7);


    errors_test=[];
    errors_train=[];

    avg_mean_errors_train = [];
    avg_mean_errors_test = [];

    for i = 1:length(cycles) 
        disp('Cycles');
        disp(cycles(i));
        
        mean_errors_test = [];
        mean_errors_train = [];


        for l = 1:2
            disp('round');
            disp(l);
            %10-fold cross validation
            fold = 10;
            indices = crossvalind('Kfold', length(input), fold); %randomly assigns indices


            for j = 1:fold
                testIdx = (indices == j); 
                trainIdx = ~testIdx;

                input_layers = size(input,2);
                output_layers = size(output,2);


                net = mlp(input_layers, hidden_nodes, output_layers, 'linear');


                options = zeros(1,20);
                options(1) = -1; %suppress warnings

                net = netopt(net, options, input(trainIdx,:), output(trainIdx,:), gradients{g});%scaled conjugate gradient/standard gradient

                [trainNet, error] = mlptrain(net, input(trainIdx,:), output(trainIdx,:), cycles(i));
                
                y = mlpfwd(trainNet, input(testIdx,:));
                y_train=mlpfwd(trainNet, input(trainIdx,:));

                error_test=(rms(sqrt((output(testIdx,1)-y(:,1)).^2 + (output(testIdx,2)-y(:,2)).^2))); 
                errors_test=[errors_test error_test];

                error_train=(rms(sqrt((output(trainIdx,1)-y_train(:,1)).^2 + (output(trainIdx,2)-y_train(:,2)).^2)));
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
shapesTrain = {'--o','-*', '--'};
shapesTest = {'-o', '--*', '-'};
figure;
for g=1:length(gradients)
    disp(gradients{g});
    plot(cycles,mean_errors_for_gradient_test{g}, shapesTrain{g}); hold on;
    plot(cycles,mean_errors_for_gradient_train{g}, shapesTest{g}); hold on;
    legendText = [legendText, strcat(gradients{g},'-test'), strcat(gradients{g},'-train')];
end

xlabel('number of epochs')
ylabel('average error')
legend(legendText);

saveas(gcf,strcat('images/rmse_epochs2','.png'));
    
