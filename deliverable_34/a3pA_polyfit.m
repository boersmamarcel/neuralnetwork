%files = {'line.mat'; 'sinus.mat'; 'irregular.mat'};
files={'line.mat'; 'sinus.mat'}

for k = 1:length(files)
    data = importdata(files{k}, '-mat');
    
    input=data.('x');
    output=data.('t');
    
    degrees=[1,2,3,4,5,6];
    
  
    
    RMStest=[];
    RMStrain=[];
 
        
    for i = 1:length(degrees);
        degree=degrees(i);
        
        final_errors_test=[];
        final_errors_train=[];
        
        for l = 1:100;
                %5-fold cross validation
        fold = 5;
        indices = crossvalind('Kfold', length(data.('x')), fold); %randomly assigns indices
        errors_test=[];
        errors_train=[];
        for j=1:fold
            testIdx = (indices == j); 
            trainIdx = ~testIdx;
            %newdata = (data.('x')-mean(data.('x')))/std(data.('x'));
            
            [p]=polyfit(data.('x')(trainIdx),data.('t')(trainIdx),degree);
            x1=linspace(min(data.('x')),max(data.('x'))); 
    
            %[p,S,mu]=polyfit(data.('x')(trainIdx),data.('t')(trainIdx),degree);
            %x1=linspace(min(newdata),max(newdata));  
            y=polyval(p,x1);
        
            error_test=sqrt((1/5)*(sum((polyval(p,data.x(testIdx))-data.('t')(testIdx)).^2))); 
            errors_test=[errors_test error_test];
            
            error_train=sqrt((1/20)*(sum((polyval(p,data.x(trainIdx))-data.('t')(trainIdx)).^2))); 
            errors_train=[errors_train error_train];
            
           
            

        end
        
         mean_errors_test=mean(errors_test);
         mean_errors_train=mean(errors_train);
    %figure;    
    %plot(x1,y); hold on;
    
    %scatter(newdata, data.t, '*');
    final_errors_test=[final_errors_test mean_errors_test];
    final_errors_train=[final_errors_train mean_errors_train];

        end    
       
    RMStest=[RMStest mean(final_errors_test)];
    RMStrain=[RMStrain mean(final_errors_train)];
    
      
  
    
    
    end
    
    figure;
    plot(degrees,RMStest); hold on;
    plot(degrees,RMStrain);
    xlabel('degrees of the polynomial')
    ylabel('RMS')
    legend('error test set','error training set')

end

