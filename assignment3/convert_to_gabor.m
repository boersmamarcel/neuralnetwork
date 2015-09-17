%%code based on http://www.mathworks.com/matlabcentral/fileexchange/23253-gabor-filter/content/Gabor%20Filter/gabor_fn.m
data = importdata('pics.mat', '-mat');
data_gabor = zeros(size(data.pics, 1), size(data.pics, 2));

for i = 1:size(data.pics,1)
    lambda  = 8;
    theta   = 0;
    psi     = [0 pi/2];
    gamma   = 0.5;
    bw      = 1;
    N       = 8;
    img_in = reshape(data.pics(1,:), 56, 46)./255; %normalize image
    img_out = zeros(size(img_in,1), size(img_in,2), N);
    for n=1:N
        gb = gabor_fn(bw,gamma,psi(1),lambda,theta)...
            + 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
        % gb is the n-th gabor filter
        img_out(:,:,n) = imfilter(img_in, gb, 'symmetric');
        % filter output to the n-th channel
        theta = theta + 2*pi/N;
        % next orientation
    end
    img_out_disp = sum(abs(img_out).^2, 3).^0.5;
    % default superposition method, L2-norm
    img_out_disp = img_out_disp./max(img_out_disp(:));
    % normalize
    data_gabor(i,:) = reshape(img_out_disp, 2576, 1);
end


save 'pics_gabor.mat' data;

