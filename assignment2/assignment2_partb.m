%initialize the model
ncentres = 2;
input_dim = 1;
mix = gmm(input_dim, ncentres, 'spherical'); mix.centres=[50; 60]; %manual initialization
% Print out the initial model
disp(' Priors Centres Variances') 
disp([mix.priors' mix.centres mix.covars'])
% Set up vector of options for EM trainer options = zeros(1, 18);
options(1) = 1; options(14) = 10;
% Prints out error values. % Max. Number of iterations.


