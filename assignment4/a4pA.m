data = importdata('pics.mat', '-mat');

glassIdx = data.classGlass == 1;
noGlassIdx = data.classGlass == 0;

classGlass = data.classGlass();

dataGlass = data.pics(glassIdx,:); %get glass data
dataNoGlass = data.pics(noGlassIdx,:); % no glass data

%construct rbf network
input_nodes = 56*46;
output_nodes = 1;
hidden_nodes = [10, 50, 100, 200];

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
            
            
            net = rbf(input_nodes, hidden_nodes(i), output_nodes, 'gaussian');
            
            options = zeros(1,14);
            options(1) = -1;
            net2 = rbfsetbf(net, options, trainData);

            net3 = rbftrain(net2, options, trainData, trainClass);

            y = rbffwd(net3, testData);
            classEst = y>0.5;

            error=(rms(classEst-testClass));

            errors = [errors error];
            
    end
    
end

mean_errors=[];
for i = 1:length(hidden_nodes);
    mean_error=mean(errors((i-1)*5+1:i*5));
    mean_errors=[mean_errors mean_error];
end

figure;
plot(hidden_nodes,mean_errors);
xlabel('number of hidden nodes')
ylabel('average error')




%choose 100 k-means
net = rbf(input_nodes, 100, output_nodes, 'gaussian');

options = zeros(1,14);
options(1) = -1;
net2 = rbfsetbf(net, options, data.pics);

net3 = rbftrain(net2, options, data.pics, data.classGlass.');

y = rbffwd(net3, data.pics);

y = y > 0.5;

M = [y data.classGlass.'];
confusionmat(M(:,1), M(:,2))
    

