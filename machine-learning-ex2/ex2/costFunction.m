function [J, grad] = costFunction(X, theta, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
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
%
%Compute log(h(X*theta))
templog(:,1) = log(sigmoid(X*theta));
%Compute log(1-h(X*theta))
templog(:,2) = log(1-(sigmoid(X*theta)));
%Compute y
tempy(:,1) = y;
%Compute (1-y)
tempy(:,2) = 1-y;

%Compute y*log(h(X*theta)) and (y-1)*log(1-h(X*theta))
temp = templog.*tempy;

%Compute cost function 
%J = (1/m)*?[-y*log(h(X*theta))-(1-y)*log(1-h(X*theta))]
J = (1/m)*(-sum(temp(:,1))-sum(temp(:,2)));

%Compute function gradient
%gradJ = (1/m)*(h(X*theta)-y)*X
grad=(1/m)*((sigmoid(X*theta)-y)'*X(:,:));
%For some reason, [h(X*theta)-y] has to be transposed. I don't know why...

%
% Note: grad should have the same dimensions as theta
%
% =============================================================
end
