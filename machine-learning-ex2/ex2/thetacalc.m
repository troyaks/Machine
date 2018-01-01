function [theta, J] = thetacalc(X, theta, y, learning_rate) %#ok<*INUSL>
[m,n] = size(X);
for i = 1:800
    [J, grad] = costFunction(X, theta, y); %#ok<*NOPRT>
    for j = 1:n
        theta(j) = theta(j) - learning_rate*grad(j);
    end
end
end
