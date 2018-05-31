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

% X is m x 3, theta is 3 x 1, h is m x 1
h = sigmoid(X * theta);

% first term, y is m x 1, log(h) is m x 1
q = sum((-1 .* y)' * log(h));

% 1-y term
q2 = sum((1 .- y)' * log(1 .- h));

% regularize it
theta(1) = 0;
sqrTheta = theta' * theta;
regCost = (lambda / (2 * m)) * sqrTheta;
J = ((1 / m) * (q - q2)) + regCost;


% gradient, (m X n)' * m x 1 =  n x 1
t = X' * (h - y);

regGrad = (lambda / m) .* theta; 

grad = ((1 / m) .* t) + regGrad;

% =============================================================

end
