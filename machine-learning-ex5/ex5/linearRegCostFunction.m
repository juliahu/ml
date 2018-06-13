function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% Init
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% cost
h = X * theta;
error = h - y;
error_sqr = error .^ 2;
q = sum(error_sqr);
J = (1/(2*m)) * q;

% gradient
grad = (1/m) * (X' * error);

% cost reg
theta(1) = 0;
thetaSum = sum(theta .^ 2);
J = J + (lambda/(2*m) * thetaSum);

% gradient reg
grad = grad + (theta * (lambda/m));

% =========================================================================

grad = grad(:);

end
