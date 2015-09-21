data = importdata('pics.mat', '-mat');

glassIdx = data.classGlass == 1;
noGlassIdx = data.classGlass == 0;

classGlass = data.classGlass();

dataGlass = data.pics(glassIdx,:); %get glass data
dataNoGlass = data.pics(noGlassIdx,:); % no glass data

fold = 10;
indicesGlass = crossvalind('Kfold', sum(glassIdx), fold);
indicesNoGlass = crossvalind('Kfold', length(data.class) - sum(glassIdx), fold);
kernels={'RBF_kernel','lin_kernel','poly_kernel'};

mean_errors_test=[];
mean_errors_train=[];



for i = 1:length(kernels);
    errors_test = [];
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



        type='c'; %classification
        kernel = kernels(i); %RBF_kernel/lin_kernel/poly_kernel
        dataprocessing = 'preprocessing'; % preprocess/original
        %GAM: regularization parameter
        % for gam low minimizing of the
        % complexity of the model is emphasized, for gam high, good fitting
        % of the training data points is stressed.
        gam = 0.01;

        if strcmp(kernel, 'poly_kernel')
            degree = 10;
            highlowbalance = 0.2;
            model = {trainData,trainClass,type,gam,[highlowbalance degree],kernel};

        elseif strcmp(kernel, 'RBF_kernel')
            sig2 = 2;
            model = {trainData,trainClass,type,gam,sig2,kernel,dataprocessing};

        else
            model = {trainData,trainClass,type,gam,[],kernel,dataprocessing};

        end

        [alpha,b] = trainlssvm(model);
        Ytest = simlssvm(model, {alpha, b}, testData);
        Ytrain = simlssvm(model, {alpha, b}, trainData);

        error_test=(rms(Ytest-testClass));
        error_train=rms(Ytrain-trainClass);

        errors_test = [errors_test error_test];
        errors_train = [errors_train error_train];
        

end

mean_errors_test=[mean_errors_test mean(errors_test)];
mean_errors_train=[mean_errors_test mean(errors_train)];



end



%figure;
%plot(errors);
%xlabel('number of hidden layers')
%ylabel('average error')
%ylim([0,1])

%give confusion matrix
confusionmat(simlssvm(model, {alpha, b}, data.pics), data.classGlass.')

