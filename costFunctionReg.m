function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%theta(1)=0;
%reg=lambda/(2*m)*(theta'*theta);
%scale=lambda/(2*m);

%reg=reg1*scale;

h=sigmoid(X*theta);%hypothesis
R=-y'*log(h);%red part 
B=(1-y)'*log(1-h);%blue part

theta(1)=0;%before adding the regularisation term we should make theta(1)=0
J=((R-B)/m)+(lambda/(2*m)*(theta'*theta));


theta(1)=0;
grad=X'*(h-y);
grad=grad./m

grad=grad+(lambda/m)*theta;


% =============================================================

end
