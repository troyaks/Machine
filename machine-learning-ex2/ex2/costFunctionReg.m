function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
%Compute log(h(X*theta))
templog(:,1) = log(sigmoid(X*theta));
%Compute log(1-h(X*theta))
templog(:,2) = log(1-(sigmoid(X*theta)));
%Compute y
tempy(:,1) = y;
%Compute (1-y)
tempy(:,2) = 1-y;
%Compute theta^2
temptheta(:,1) = theta.^2;
%Compute y*log(h(X*theta)) and (y-1)*log(1-h(X*theta))
temp = templog.*tempy;
%Compute cost function 
%J = (1/m)*SUM[-y*log(h(X*theta))-(1-y)*log(1-h(X*theta))] + 
%       (lambda/2m)*SUM[theta^2]
J = (1/m)*(-sum(temp(:,1))-sum(temp(:,2))) + ...
    (lambda/(2*m))*sum(temptheta(2:end,1));
%Compute function gradient regularized
%gradJ = (1/m)*(h(X*theta)-y)'*X + (lambda/m)*theta
grad = (1/m)*((sigmoid(X*theta)-y)'*X(:,:))+(lambda/m)*(theta');
%Note that in the gradient regularized case we do not regularize theta zero
grad(1) = grad(1)-(lambda/m)*theta(1);
%For some reason, [h(X*theta)-y] has to be transposed. Actually,
%if you do not transpose it your grad doesn't compute.

end
