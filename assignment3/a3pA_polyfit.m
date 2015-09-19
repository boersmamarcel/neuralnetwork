files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};

for k = 1:length(files)
    data = importdata(files{k}, '-mat');
    
    input=data.('x');
    output=data.('t');
    
    degrees=[1,2,3,4,5,6,7];
    errors_test=[];
    errors_train=[];
    
    for i = 1:length(degrees);
        degree=degrees(i);
        
                %5-fold cross validation
        fold = 5;
        indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
        for j=1:fold
            testIdx = (indices == j); 
            trainIdx = ~testIdx;
            newdata = (data.('x')-mean(data.('x')))/std(data.('x'));
    
            [p,S,mu]=polyfit(newdata(trainIdx),data.('t')(trainIdx),degree);
            x1=linspace(min(newdata),max(newdata));  
            y=polyval(p,x1);
        
            error_test=(rms(data.('t')(testIdx)-y(testIdx).')); 
            errors_test=[errors_test error_test];
            
            error_train=(rms(data.('t')(trainIdx)-y(trainIdx).')); 
            errors_train=[errors_train error_train];

        end
        
    figure;    
    plot(x1,y); hold on;
    
    scatter(newdata, data.t, '*');
    

    end
    
    mean_errors_test=[];
    mean_errors_train=[];
    for i = 1:length(degrees);
        mean_error_test=mean(errors_test((i-1)*fold+1:i*fold));
        mean_errors_test=[mean_errors_test mean_error_test];
        mean_error_train=mean(errors_train((i-1)*fold+1:i*fold));
        mean_errors_train=[mean_errors_train mean_error_train];
    end
    
    figure;
    plot(degrees,mean_errors_test); hold on;
    plot(degrees,mean_errors_train);
    xlabel('degrees of the polynomial')
    ylabel('average error')
    legend('error test set','error training set')
    
    
end
