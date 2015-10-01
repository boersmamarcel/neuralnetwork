
errors=[];
neurons=[2,5,8,11,14,17,20];
for k = 1:length(neurons);    
    disp(k)

    X = exprnd(2,[1,500]);
    X=1-(X/max(X));
    Y=randn(500,1);
    Y=Y.';
    %X = rand(1,1000);
    %Y = rand(1,1000);
    %Y=Y/max(Y)
    %figure;
    M=[X; Y];
    scatter(M(1,:),M(2,:),'bo'); hold on; 


    xneurons=neurons(7);
    yneurons=neurons(7);
    totalneurons=xneurons*yneurons;
    net=som(2,[xneurons,yneurons]);

    options = foptions;
    options(14) = 1000 ; %epochs ;
    options(18) = 0.5;   %initial_learning_rate;  
    options(16) = 0.05;  %final_learning_rate; 
    options(17) = 8;     %Initial neighbourhood size
    options(15) = 1;     %Final neighbourhood size
    
    disp('ordening phase')

    net2=somtrain(net,options,M.');
    c2 = sompak(net2);
    plot(c2(:, 1), c2(:, 2), 'r*'); hold on;


    options(14) = 500*xneurons*yneurons;  %epochs
    options(18) = 0.01; %initial_learning_rate; 
    options(16) = 0.01; %final_learning_rate;
    options(17) = 1;    %Initial neighbourhood size
    options(15) = 0;    %Final neighbourhood size
    
    disp('convergence phase')
    net3 = somtrain(net2, options, M.');
    c3 = sompak(net3);

    plot(c3(:, 1), c3(:, 2), 'g^');
    legend('data','neurons after ordering phase','neurons after convergence phase')

    figure;
    for i = 1:totalneurons;

        neighbours = find(net3.inode_dist(:,:,i) == 1);
        for j = 1:length(neighbours);
            index = neighbours(j);
            plot([c3(i,1),c3(index,1)],[c3(i,2) c3(index,2)],'g'); hold on;
        end

    end

    error=0;

    for i =1:length(M);
        distances=[];       
        for j = 1:length(c3);
            distance=sqrt((c3(j,1)-M(1,i))^2+(c3(j,2)-M(2,i))^2);
            distances=[distances distance];
        end

        err = min(distances);
        error=[error err]; 


    end

    mean_error=mean(error);
    errors=[errors mean_error];
end
