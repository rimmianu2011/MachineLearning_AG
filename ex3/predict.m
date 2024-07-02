function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
% Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
% Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
% X = reshape(sin(1:16), 8, 2);

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

column_ones = ones(m, 1);
a1 = [column_ones, X];

z2 = a1 * Theta1';
sigmoid_val = sigmoid(z2);

size_sigmoid = size(sigmoid_val, 1);
sigmoid_ones = ones(size_sigmoid, 1);
a2 = [sigmoid_ones, sigmoid_val];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

[~, p]= max(a3, [], 2);

% p = predict(Theta1, Theta2, X)






% =========================================================================


end
