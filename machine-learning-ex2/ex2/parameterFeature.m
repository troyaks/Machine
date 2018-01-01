function X = parameterFeature(X)
%Function to enclose the high and low values of a data between 0.1 and 10.

[m,n] = size(X);
k = ones(1,n);
for i = 1:n
    while max(X(:,i))>1
        k(1,i)=10*k(1,i);
        X(:,i)=X(:,i)/10;
    end
end
end
