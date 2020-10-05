X = [ones(3,1) magic(3)];
y = [1 0 1]';
theta = [-2 -1 1 2]';

% un-regularized
%[j g] = costFunction(theta, X, y)
% or...
[j g] = costFunctionReg(theta, X, y, 0)

% results
%j = 4.6832



% regularized
[j g] = costFunctionReg(theta, X, y, 4)

