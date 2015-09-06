%creating the dataset
x=[1:9]; 
C1=[2; 5; 7; 6; 4; 2; 1; 0; 0];
C2=[0; 0; 0; 1; 3; 3; 5; 4; 2];
M=[x.' C1 C2];

%Making the barplot with frequencies
figure 
bar(x,M(:,2:3))
legend({'C1','C2'})
xlabel('X')
ylabel('frequencies')

%Making a barplot with conditional probabilities
total_inst_X=[]
for i = 1:length(C1)
    total=sum(M(i,2:3))
    total_inst_X(i)=total
end
M=[M total_inst_X.'];

figure
bar(x,[M(:,2)./total_inst_X.' M(:,3)./total_inst_X.'])
legend({'C1','C2'})
xlabel('X')
ylabel('conditional probabilities')

%calculating the misclassifications 
misclassifications=[];
for split = 1:length(x)-1
    misclassifications(split)=sum(C1(split+1:length(C1)))+sum(C2(1:split))      
end

%Finding optimal decision boundary when costs are equal
best_split1=find(misclassifications==min(misclassifications))

%Finding optimal decision boundary when L_ba=2 and L_ab=1
split=1;
L_ba=2;
L_ab=1;
cost_C1=1;
cost_C2=2;
while cost_C1<cost_C2
    cost_C1=L_ab*(C2(split)/(C1(split)+C2(split)));
    cost_C2=L_ba*(C1(split)/(C1(split)+C2(split)));
    if cost_C1<cost_C2
        split = split+1;
    else
        break
    end
end

best_split2=split

   
   




    
    
