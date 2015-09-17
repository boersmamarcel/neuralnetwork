function net = rbfsetbf(net, options, x)
%RBFSETBF Set basis functions of RBF from data.
%
%	Description
%	NET = RBFSETBF(NET, OPTIONS, X) sets the basis functions of the RBF
%	network NET so that they model the unconditional density of the
%	dataset X.  This is done by training a GMM with spherical covariances
%	using GMMEM.  The OPTIONS vector is passed to GMMEM. The widths of
%	the functions are set by a call to RBFSETFW.
%
%	See also
%	RBFTRAIN, RBFSETFW, GMMEM
%

%	Copyright (c) Ian T Nabney (1996-2001)

errstring = consist(net, 'rbf', x);
if ~isempty(errstring)
  error(errstring);
end

% Initialise the parameters from the input data
% Just use a small number of k means iterations
kmoptions = zeros(1, 18);
kmoptions(1) = -1;	% Turn off warnings
kmoptions(14) = 50;  % 50 iterations should do the trick

% Try a simple variant of k-means clustering....

clusters = net.nhidden;
centres = kmeans(rand(clusters,size(x,2)), x, kmoptions);

% Now set the centres of the RBF from the centres of the mixture model
net.c = centres;

% options(7) gives scale of function widths
net = rbfsetfw(net, options(7));