data = importdata('pics_gabor.mat', '-mat');

glassIdx = data.classGlass == 1;
noGlassIdx = data.classGlass == 0;

classGlass = data.classGlass();

dataGlass = data.pics(glassIdx,:); %get glass data
dataNoGlass = data.pics(noGlassIdx,:); % no glass data

%construct rbf network
input_nodes = 56*46;
output_nodes = 1;
hidden_nodes = [100,500];

RMStest=[];
RMStrain=[];

for i = 1:length(hidden_nodes)
    final_errors_test=[];
    final_errors_train=[];
    fprintf('Neural network with hidden nodes:%d\n', hidden_nodes(i));
    
    for l=1:10
    fold = 10;
    indicesGlass = crossvalind('Kfold', sum(glassIdx), fold);
    indicesNoGlass = crossvalind('Kfold', length(data.class) - sum(glassIdx), fold);
    
    errors_test=[];
    errors_train=[];
    
    for j = 1:fold

            testIdxGlass = (indicesGlass == j); 
            testIdxNoGlass = (indicesNoGlass == j);
            trainIdxGlass = ~testIdxGlass;
            trainIdxNoGlass = ~testIdxNoGlass;
            
            testData = [dataGlass(testIdxGlass,:); dataNoGlass(testIdxNoGlass,:)];
            trainData = [dataGlass(trainIdxGlass,:); dataNoGlass(trainIdxNoGlass,:)];
            
            testClass = [ones(length(dataGlass(testIdxGlass)),1); zeros(length(dataNoGlass(testIdxNoGlass)),1)];
            trainClass = [ones(length(dataGlass(trainIdxGlass)),1); zeros(length(dataNoGlass(trainIdxNoGlass)),1)];
            
            
            net = mlp(input_nodes, hidden_nodes(i), output_nodes, 'logistic');
            [trainNet, errorMLP] = mlptrain(net, trainData, trainClass, 10);

             y = mlpfwd(trainNet, testData);
             y = y>0.5;
             ytrain=mlpfwd(trainNet,trainData);
             ytrain=ytrain>0.5;

             error_test=(rms(testClass-y));
             error_train=rms(trainClass-ytrain);

             errors_test = [errors_test error_test];
             errors_train = [errors_train error_train];
            
    end
    
    final_errors_test=[final_errors_test mean(errors_test)];
    final_errors_train=[final_errors_train mean(errors_train)];
    
    end
    
    RMStest=[RMStest mean(final_errors_test)];
    RMStrain=[RMStrain mean(final_errors_train)];
    
    
end

figure;
    plot(hidden_nodes,RMStest); hold on;
    plot(hidden_nodes,RMStrain);
    xlabel('number of hidden layers')
    ylabel('RMS')
    legend('error test set','error training set')
    ylim([0 1])

    
%pick 500 hidden nodes
net = mlp(2576, 3000, 1, 'logistic');
[trainNet, errorMLP] = mlptrain(net, data.pics, data.classGlass.', 40);

y = mlpfwd(trainNet, data.pics);
y = y > 0.5;

M = [y data.classGlass.'];
confusionmat(M(:,1), M(:,2))
    
