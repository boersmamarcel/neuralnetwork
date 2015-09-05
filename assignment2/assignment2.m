%probabilities c_1 and c_2


%prior distributions
prior_c1 = 27/45; %count data-points in c1 vs all data-points (see slides)
prior_c2 = 18/45;

%Class conditionals

% P(x | c1) and P(X | c2)
%this is the joint probability p(x,c) however this looks similar to the
%plots on the slides
px_c1 = [0 2 5 7 6 4 2 1 0 0 0 0]/45;
px_c2 = [0 0 0 0 1 3 3 5 4 2 0 0]/45;

figure;
stairs(px_c1); hold on;
stairs(px_c2);

% P(x | c1) and P(X | c2)
% this is the class conditional probability but it does not look similar to
% the histogram on the slides however, we are required to calculate this
% probability
px_c1 = [0 2 5 7 6 4 2 1 0 0 0 0]/27;
px_c2 = [0 0 0 0 1 3 3 5 4 2 0 0]/18;

figure;
stairs(px_c1); hold on;
stairs(px_c2);

%p(x) = p(x | c1 ) P(C1) + p(x|c2)P(c2)
px = px_c1*prior_c1 + px_c2*prior_c2;

% posterior P(c | x) = p(x|c)p(c) / p(x) 
posterior_c1 = px_c1*prior_c1./px; % notice the element wise division
posterior_c2 = px_c2*prior_c2./px;

figure;
stairs(posterior_c1); hold on;
stairs(posterior_c2);