files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};

for k = 1:length(files)
    data = importdata(files{k}, '-mat');
    
    input=data.('x');
    output=data.('t');
    
    degrees=[2,4,6]
    errors=[]
    
    for i = 1:length(degrees)
        degree=degrees(i)
        
                %5-fold cross validation
        fold = 5;
        indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
        for j=1:fold
            testIdx = (indices == j); 
            trainIdx = ~testIdx;
    
            p=polyfit(data.('x')(trainIdx),data.('t')(trainIdx),degree);
            x1=linspace(0,max(data.('x')));  
            y=polyval(p,x1);
        
            error=(rms(data.('t')(testIdx)-y(testIdx).')); 
            errors=[errors error];

        end
        
    figure;    
    plot(x1,y); hold on;
    scatter(data.('x'), data.('t'));

    end
    
    mean_errors=[];
    for i = 1:length(degrees);
        mean_error=mean(errors((i-1)*fold+1:i*fold));
        mean_errors=[mean_errors mean_error];
    end
    
    figure;
    plot(degrees,mean_errors);
    xlabel('degrees of the polynomial')
    ylabel('average error')
    
    
end
