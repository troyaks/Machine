function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%This cost fuction comes to calculate a neural network with 1 hidden
%layer which has 25 neurons plus one. It has 400 inputs plus 1 and 10 
%outputs. 

%Build the Y matrix of results
I = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
  Y(i,:) = I(y(i),:);
end
%Feeding foward the Neural Network with Thetas already given
a1=[ones(m,1), X]; %Input parameters
% ---------------- Start of Hidden layers 
z2 = a1*Theta1'; %Calculate z2
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1),a2]; %First one with byas
% ---------------- End of Hidden layers
z3 = a2*Theta2'; %Calculate z3
a3 = sigmoid(z3); %Output parameter from Output layer
%Calculate cost function
%Sum over m training values and then sum over K labels in output layer
costNoReg = (1/m)*sum(sum((-Y.*log(a3)-(1-Y).*log(1-a3))));
%Calculate regularization
%Sum over K labels and then sum over x parameters from each layer
%Each theta has its own K labels and x parameters respectively
%Note that we do not regularize theta zero
sumt1 = sum(sum(Theta1(:, 2:end).^2));
sumt2 = sum(sum((Theta2(:, 2:end).^2)));
Regularization = (lambda/(2*m))*(sumt1 + sumt2);
%Get regularized Cost function for the Neural Network
J = costNoReg + Regularization;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
