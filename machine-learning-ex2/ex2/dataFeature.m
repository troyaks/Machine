function X = dataFeature(X,degree)
%A function to call another functions and give a final value.
X = parameterFeature(mapFeature(X(:,1),X(:,2),degree));
end