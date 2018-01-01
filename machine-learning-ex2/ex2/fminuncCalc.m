function [theta,cost,exitflag,output,grad] = ... 
    fminuncCalc(X,y,lambda,initial_theta)

%Set options for fminunc
options = optimoptions('fminunc');
options = optimoptions(options,'MaxIterations',1000);
options = optimoptions(options,'FunctionTolerance',1e-9);
options = optimoptions(options,'StepTolerance', 1e-9);
options = optimoptions(options,'Algorithm', 'trust-region');
options = optimoptions(options,'SpecifyObjectiveGradient', true);
options = optimoptions(options,'Hessian', 'off');
%options = optimoptions(options,'PlotFcn',@optimplotfval);
options = optimoptions(options,'FiniteDifferenceType','central');
options = optimoptions(options,'OptimalityTolerance',1e-9);
%Set parameters for fminunc
t = 0;
%Compute Theta and Function Value
[theta,cost,exitflag,output,grad] = ...
    fminunc(@(t)(costFunctionReg(t,X,y,lambda)),initial_theta,options);
fprintf('Exit flag: %f \n', exitflag);
fprintf('Gradient of each theta: \n');
fprintf('%f \n',grad);
fprintf('Cost at theta found by fminunc: %f \n', cost);
fprintf('Theta values found by fminunc: \n');
fprintf(' %f \n', theta);