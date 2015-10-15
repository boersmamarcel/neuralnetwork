%[X,T] = simpleseries_dataset;
net = layrecnet(1:2,10);
%[Xs,Xi,Ai,Ts] = preparets(net,X,T);
%net = train(net,Xs,Ts,Xi,Ai);
%view(net)
%Y = net(Xs,Xi,Ai);
%perf = perform(net,y,Ts)

%calculating distances to source
max_y = 800;
source = [542.0, max_y-439.0];
distance_to_source=[];
for i=1:length(data(:,1));
    
    d=sqrt( (data(i,1)-source(1))^2 + (data(i,2)-source(2))^2 );
    distance_to_source=[distance_to_source d];
    
end
distance_to_source=distance_to_source(1:46);
X=con2seq(distance_to_source);

%targets consist of the direction
targets=[]
for i=1:length(data(:,1))-1;
    
    dx=data(i+1,1)-data(i,1);
    dy=data(i+1,2)-data(i,2);
    t=[dx];
    targets=[targets t];
    
end

Y=con2seq(targets);


[Xs,Xi,Ai,Ts,EWs,shift] = preparets(net,X,Y);
net = train(net,Xs,Ts,Xi,Ai);
y = net(Xs,Xi,Ai);
y = net(Xs,Xi,Ai);