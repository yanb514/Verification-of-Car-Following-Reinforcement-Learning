load ACASXU_run2a_2_9_batch_2000.mat;
Layers = [];
n = length(b);
for i=1:n - 1
    bi = cell2mat(b(i));
    Wi = cell2mat(W(i));
    Li = Layer(Wi, bi, 'ReLU');
    Layers = [Layers Li];
end
bn = cell2mat(b(n));
Wn = cell2mat(W(n));
Ln = Layer(Wn, bn, 'Linear');

Layers = [Layers Ln];
F = FFNN(Layers);

% Input Constraints (see paper Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks, CAV 2017)
% 0 <= i1 (\rho) <= 60760,  
% -3.141592 <= i2 (\theta) <= -0.75*3.141592  
% -0.1 <= i3 (\shi) <= 0.1
% 600 <= i4 (\v_own) <= 1200,
% 600 <= i5 (\v_in) <= 1200

lb = [0; -3.141592; -0.1; 600; 600];
ub = [60760; -0.75*3.141592; 0.1; 1200; 1200];

% normalize input
for i=1:5
    lb(i) = (lb(i) - means_for_scaling(i))/range_for_scaling(i);
    ub(i) = (ub(i) - means_for_scaling(i))/range_for_scaling(i);   
end


V = Reduction.getVertices(lb, ub);

I = Polyhedron('V', V');

[R, t] = F.reach(I, 'exact', 4, []); % exact reach set
save F.mat F; % save the verified network
F.print('F.info'); % print all information to a file
