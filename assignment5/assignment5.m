data = importdata('pics.mat');


%% Show image for different number of components

PCAN = [2576, 1500, 500, 200, 100, 50, 10, 5, 1];
[PCcoeff, PCVec] = pca(data.pics);


for i=1:length(PCAN)
    N = PCAN(i);

    finalData = PCVec(:,1:N).'*data.pics.';

    back = finalData.'*PCVec(:,1:N).';

    figure;
    imagesc(reshape(back(200,:),56,46));
    saveas(gcf, strcat('face_', num2str(N), '.png'));
    
end


%% Errors

errors = [];

for i=1:length(PCAN)
    N = PCAN(i);

    finalData = PCVec(:,1:N).'*data.pics.';

    back = finalData.'*PCVec(:,1:N).';

    errors = [errors distance(data.pics, back)];
end
    
    
figure;
title('Error for differnt number of PCA components');
plot(PCAN, errors);
ylabel('RMSE');
xlabel('N-components');

saveas(gcf, strcat('error','.png'));