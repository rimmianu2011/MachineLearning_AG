function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
data = load('ex1data1.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); % number of training examples
n = length(X); % number of training examples

X = [ones(n,1),data(:,1)]; % Add a column of ones to x

theta = zeros(2, 1); % initialize fitting parameters
num_iters = 1500;
alpha = 0.01;
J_history = zeros(num_iters, 1);

% the following 2 lines were executed to check if the computeCost function 
% was running correctly.
% computeCost(X, y, theta);
% computeCost(X, y, [-1; 2]);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % calculates the value of hypothesis function.     
    hypo_val = X*theta;
    errors = hypo_val - y;
    % calculates the partial derivative part of the gradient descent.     
    delta = (1/m) * (X' * errors);
    % updates the value of theta after every iteration.     
    theta = theta - alpha * delta;




    % ============================================================

    % Save the cost J in every iteration    
    % calls the computeCost function which calculates the value of 
    % J theta function.     
    J_history(iter) = computeCost(X, y, theta);

end

end
