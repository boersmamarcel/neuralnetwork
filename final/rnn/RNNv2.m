max_y = 800;
source = [542.0, max_y-439.0];
%[X,T] = simpleseries_dataset;
net = layrecnet(1:2,10);
%[Xs,Xi,Ai,Ts] = preparets(net,X,T);
%net = train(net,Xs,Ts,Xi,Ai);
%view(net)
%Y = net(Xs,Xi,Ai);
%perf = perform(net,y,Ts)

%calculating distances to source
distance_to_source=[];
for i=1:length(data(:,1));
    
    d=sqrt( (data(i,1)-source(1))^2 + (data(i,2)-source(2))^2 );
    distance_to_source=[distance_to_source d];
    
end
distance_to_source=distance_to_source(1:46);
X=con2seq(distance_to_source);

%calculating dx and dy to source
loc_to_source=[];
for i=1:length(data(:,1));
    
    dx=data(i,1)-source(1);
    dy=data(i,2)-source(2);
    loc_to_source=[loc_to_source [dx;dy]];
    
end
loc_to_source=loc_to_source(:,1:46);
X2=con2seq(loc_to_source);

input=[distance_to_source; loc_to_source];
X3=con2seq(input);


%targets consist of the direction
targets=[];
for i=1:length(data(:,1))-1;
    
    dx=data(i+1,1)-data(i,1);
    dy=data(i+1,2)-data(i,2);
    t=[dx;dy];
    targets=[targets t];
    
end

Y=con2seq(targets);

net=train(net,X3,Y);
net=train(net,input,targets)

%[Xs,Xi,Ai,Ts,EWs,shift] = preparets(net,X2,Y);
%net = train(net,Xs,Ts);
y = net(X3);
perf = perform(net,y,Y);
%y = net(Xs,Xi,Ai);

%to plot:
y=cell2mat(y);
figure;
scatter(targets(1,:),targets(2,:),'b'); hold on;
scatter(y(1,:),y(2,:),'r');