function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
% data = load('ex1data1.txt');
% X = data(:, 1);
% y = data(:, 2);
% plotData(X,y);
% theta = zeros(2,1);
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta = [0, 0];

% pred = theta(1) + X * theta(2);
pred = X * theta;
square_errors = (pred - y).^2;
j_val = 1/(2*m);
J = j_val * sum(square_errors);




% =========================================================================

end
