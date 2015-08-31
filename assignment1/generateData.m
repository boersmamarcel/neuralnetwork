function [x,y,class]=generateData(steps, r, offsetX, offsetY, class, sigma, angle)
    x = -1:2/(steps-1):1
    y = class*sqrt(r - x.^2)
    
    x = x + offsetX + normrnd(0,sigma,1,steps)
    y = y + offsetY + normrnd(0,sigma,1,steps)
    
    %rotation
    rot = [cosd(angle) -sind(angle); sind(angle) cosd(angle)]*[x;y];
    
    x = rot(1,:);
    y = rot(2,:);
    
    class = class.*ones(1,500)