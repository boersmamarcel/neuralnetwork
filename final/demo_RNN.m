time = 1:10:7200;
timeh = time(1:end/2);
timenh = time(end/2 +1 : end);
input = sind(time(1:end/2));
toPredict = sind(time(end/2 +1 : end));
% plot(time(1:end/2),input,'o-g',time(end/2+1:end), toPredict,'+-r')
% legend('input to neural network','expected output');
inputSeries = tonndata(timeh,true,false);
targetSeries = tonndata(input,true,false);

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:10;
feedbackDelays = 1:10;
hiddenLayerSize = 50;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize)

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer states.
% Using PREPARETS allows you to keep your original time series data unchanged, while
% easily customizing it for networks with differing numbers of delays, with
% open loop or closed loop feedback modes.
[inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);