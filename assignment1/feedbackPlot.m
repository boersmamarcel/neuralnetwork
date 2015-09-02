function [] = feedbackPlot(X,Y, group, alpha, beta)
yline=alpha+beta*X;
figure
    gscatter(X,Y,group); hold on;
    plot(X,yline);
   