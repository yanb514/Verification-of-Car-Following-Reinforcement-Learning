clear y yp x x1 x2 x3
close all

W1 = [1 -1 0.5; 2 -1 1]; % 2x3
b1 = [-1; 0.5]; 
W2 = [-2 1; 0 1; -2 -2; 3 -1];  % 4x2 
b2 = [1;3;-2;-1];
W3 = [1 2 3 4]; %1x4
b3 = [2];
L1 = LayerS(W1, b1, 'poslin'); % sigmoid, purelin, poslin, tanh
L2 = LayerS(W2, b2, 'tanh');
L3 = LayerS(W3, b3, 'sigmoid');

F = FFNNS([L1 L2 L3]); % construct an NNV FFNN
lb = [-1; -2; 0]; % lower bound vector
ub = [1; 1; 2]; % upper bound vector

n = 30;

test_ind = 2;

switch test_ind
    case 1
        x1 = linspace(lb(1),ub(1),n);
        x2 = 0.5 * ones(size(x1));
        x3 = 1 * ones(size(x1));     
    case 2
        x2 = linspace(lb(2),ub(2),n);
        x1 = -0.5 * ones(size(x2));
        x3 = 1 * ones(size(x2));
    case 3
        x3 = linspace(lb(3),ub(3),n);
        x2 = 0.5 * ones(size(x3));
        x1 = -0.5 * ones(size(x3));
end

X = [x1;x2;x3];

test_var = X(test_ind,:);
for i = 1:numel(test_var)
    yp(i,:)=F.gradient(X(:,i));
    y(i)=F.evaluate(X(:,i));
end

figure('Position',[100 100 400 200])
subplot(121)
plot(test_var,y,'LineWidth',2);
xlabel(sprintf('x%d',test_ind)); ylabel('f(x)')
title('input - output')

subplot(122)
plot(test_var,yp(:,test_ind),'LineWidth',2);
xlabel(sprintf('x%d',test_ind)); ylabel('f prime(x)')
title('input - output gradient')

