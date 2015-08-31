function [x,y,class]=generateData(steps, r, offsetX, offsetY, class, sigma)
    x = -1:2/(steps-1):1
    y = class*sqrt(r - x.^2)
    
    x = x + offsetX + normrnd(0,sigma,1,steps)
    y = y + offsetY + normrnd(0,sigma,1,steps)
    
    class = class.*ones(1,500)