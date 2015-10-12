max_y = 800;
source = [542.0, max_y-439.0];
net = layrecnet(1:2,40);
%[Xs,Xi,Ai,Ts] = preparets(net,X,T);
%net = train(net,Xs,Ts,Xi,Ai);
%view(net)
%Y = net(Xs,Xi,Ai);
%perf = perform(net,y,Ts)

X=cell(1,agents);
Y=cell(1,agents);

%{
for i = 1:agents;
    
    % selecting features ind 1
    indloc=find(featureMatrix(:,1)==i);
    input=featureMatrix(indloc,[2:5 8:9]).';
    input=input(:,1:46);
    target=featureMatrix(indloc,[6:7]).';
    target=target(:,1:46);
    
    %adding information ind 1 to C
    X{i}=input;
    Y{i}=target;
end
%}

X=cell(1,46);
Y=cell(1,46);

for i = 1:46;
    
    input=featureMatrix((i-1)*35+1:i*35,[2:5 8]).';
    
    target=featureMatrix((i-1)*35+1:i*35,[6:7]).';
    
    
    X{i}=input;
    Y{i}=target;
end





net.trainParam.max_fail=8
net=train(net,X,Y);
%net=train(net,input,targets)

%[Xs,Xi,Ai,Ts,EWs,shift] = preparets(net,X2,Y);
%net = train(net,Xs,Ts);
y = net(X);
%perf = perform(net,y,Y);
%y = net(Xs,Xi,Ai);
%test
%to plot:
y=cell2mat(y);
Y2=cell2mat(Y)
figure;
scatter(Y2(1,:),Y2(2,:),'b'); hold on;
scatter(y(1,:),y(2,:),'r');