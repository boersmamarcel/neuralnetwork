[a,b,class1]=generateData(500,1,2,2.1,1,0.2)
[c,d,class2]=generateData(500,1,2,1.9,-1,0.2)
w0=ones(1,500)
A=[w0;a;b;class1]
B=[w0;c;d;class2]
M=[A B]
M=M.'
input=M(:,1:3)
output=M(:,4)
weights=perceptronConvergenceAlgorithm(input, output, [0 0 0], 0.1, 100)

alpha=-weights(1)/weights(3)
beta=-weights(2)/weights(3)
feedbackPlot(M(:,2),M(:,3),M(:,4),alpha,beta)
