data = importdata('pics.mat', '-mat');

glassIdx = data.classGlass == 1;
noGlassIdx = data.classGlass == 0;

classGlass = data.classGlass();

dataGlass = data.pics(glassIdx,:); %get glass data
dataNoGlass = data.pics(noGlassIdx,:); % no glass data

%construct rbf network
input_nodes = 56*46;
output_nodes = 1;
hidden_nodes = [100, 500, 1000, 1500];

errors = [];

for i = 1:length(hidden_nodes)
    fprintf('Neural network with hidden nodes:%d\n', hidden_nodes(i));
    fold = 10;
    indicesGlass = crossvalind('Kfold', sum(glassIdx), fold);
    indicesNoGlass = crossvalind('Kfold', length(data.class) - sum(glassIdx), fold);
    
    for j = 1:fold

            testIdxGlass = (indicesGlass == j); 
            testIdxNoGlass = (indicesNoGlass == j);
            trainIdxGlass = ~testIdxGlass;
            trainIdxNoGlass = ~testIdxNoGlass;
            
            testData = [dataGlass(testIdxGlass,:); dataNoGlass(testIdxNoGlass,:)];
            trainData = [dataGlass(trainIdxGlass,:); dataNoGlass(trainIdxNoGlass,:)];
            
            testClass = [ones(length(dataGlass(testIdxGlass)),1); zeros(length(dataNoGlass(testIdxNoGlass)),1)];
            trainClass = [ones(length(dataGlass(trainIdxGlass)),1); zeros(length(dataNoGlass(trainIdxNoGlass)),1)];
            
            
            net = mlp(input_nodes, hidden_nodes(i), output_nodes, 'linear');
            [trainNet, errorMLP] = mlptrain(net, trainData, trainClass, 40);

             y = mlpfwd(trainNet, testData);
             y = y>0;

             error=(rms(testClass-y));

             errors = [errors error];
            
    end
    
end

mean_errors=[];
for i = 1:length(hidden_nodes);
    mean_error=mean(errors((i-1)*10+1:i*10));
    mean_errors=[mean_errors mean_error];
end

figure;
plot(hidden_nodes,mean_errors);
xlabel('number of hidden layers')
ylabel('average error')

