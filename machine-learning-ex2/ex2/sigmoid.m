function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.
%   z is Theta transposed times x.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[m,n] = size(z);    % Getting rows and columns of 'z'

for j = 1:n
    for i = 1:m
    g(i,j) = 1/(1+exp(-z(i,j))); %Sigmoid formula
    end
end
    


% =============================================================

end
