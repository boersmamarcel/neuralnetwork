function [x,y,class]=generateData(steps, r, offsetX, offsetY, class, sigma, angle)
    x = -1:2/(steps-1):1;
    y = sqrt(r - x.^2);
    
    noise = normrnd(0,sigma,1,steps);
    
    v=acos(x);
    
    disp(offsetX);
    disp(offsetY);
    
    x = x + offsetX + (cos(v)).*noise;
    y = y + class*offsetY + (sin(v)).*noise;
    
    %rotation
    rot = [cosd(angle) -sind(angle); sind(angle) cosd(angle)]*[x;y];
    
    x = rot(1,:);
    y = rot(2,:);
        
    y = y*class;
    class = class.*ones(1,length(x));