%probabilities c_1 and c_2


%prior distributions
prior_c1 = 27/45; %count data-points in c1 vs all data-points (see slides)
prior_c2 = 18/45;

%Class conditionals

% P(x | c1) and P(X | c2)
%this is the joint probability p(x,c) however this looks similar to the
%plots on the slides
px_c1 = [2 5 7 6 4 2 1 0 0]/45;
px_c2 = [0 0 0 1 3 3 5 4 2]/45;

x=[1:10].';
figure;
stairs(x,[[px_c1 0].' [px_c2 0].']); 
legend({'C1','C2'});
xlabel('X');
ylabel('relative frequencies');


% P(x | c1) and P(X | c2)
% this is the class conditional probability but it does not look similar to
% the histogram on the slides however, we are required to calculate this
% probability
px_c1_cond = [2 5 7 6 4 2 1 0 0]/27;
px_c2_cond = [0 0 0 1 3 3 5 4 2]/18;

%p(x) = p(x | c1 ) P(C1) + p(x|c2)P(c2)
px = px_c1_cond*prior_c1 + px_c2_cond*prior_c2;

% posterior P(c | x) = p(x|c)p(c) / p(x) 
posterior_c1 = px_c1_cond*prior_c1./px; % notice the element wise division
posterior_c2 = px_c2_cond*prior_c2./px;

figure;
stairs(x,[[posterior_c1 0].' [posterior_c2 0].']);
legend({'C1','C2'});
xlabel('X');
ylabel('conditional probabilities');

%plot errors for different decision boundaries
errors = [];
for i = 1:9 
    t1 = px_c1_cond*prior_c1;
    t2 = px_c2_cond*prior_c2;
    
    p_error = sum(t1((i+1):9)) + sum(t2(1:i));
    
    errors = [errors p_error];
end


%weighted costs, optimal decision boundary shifts to the right
errorsWeighted = [];
for i = 1:9 
    t1 = px_c1_cond*prior_c1;
    t2 = px_c2_cond*prior_c2;
    
    p_error = sum(t1((i+1):9)) + sum(t2(1:i))*2;
    
    errorsWeighted = [errorsWeighted p_error];
end

figure;
plot(errors); hold on;
plot(errorsWeighted);
legend('Errors', 'Weighted');
