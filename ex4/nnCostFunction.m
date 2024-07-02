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


% input_layer_size = 2;              % input layer
% hidden_layer_size = 2;              % hidden layer
% num_labels = 4;              % number of labels
% nn_params = [ 1:18 ] / 10;  % nn_params
% X = cos([1 2 ; 3 4 ; 5 6]);
% y = [4; 2; 3];
% lambda = 4;


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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
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

y_matrix = eye(num_labels);
y_matrix = y_matrix(y,:);

J = (1/m) * sum(sum((-y_matrix .* log(a3)) - (1-y_matrix) .* log(1-a3)));

% disp(J)
delta1 = 0;
delta2 = 0;
for i = 1:m
    a1_i = a1(i, :);
    a2_i = a2(i, :);
    a3_i = a3(i, :);

    y_i = y_matrix(i, :);

    d3 = a3_i - y_i;
%     disp(sigmoidGradient(z2))
    d2 = (Theta2(:,2:end)' * d3')' .* sigmoidGradient(z2(i,:));
%     disp(d2)

    delta1 = delta1 + (d2' * a1_i);
    delta2 = delta2 + (d3' * a2_i);
end

% disp(delta2)
Theta1_grad = (1/m) * delta1;
Theta2_grad = (1/m) * delta2;

regularization = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularization;
% disp(J)

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(: ,2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(: ,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% disp(grad)
end
