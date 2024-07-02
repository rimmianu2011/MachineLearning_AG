function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


load('ex6data3.mat');
% 
% % Plot training data
% plotData(X, y);

% x1plot = linspace(-2, 2, 10)';
% x2plot = linspace(-2, 2, 10)';
% [X1, X2] = meshgrid(x1plot, x2plot);
% X = [X1(:) X2(:)];
% Xval = X + 0.3;
% y = double(sum(exp(X),2) > 3);
% yval = double(sum(exp(Xval),2) > 3);

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = zeros(length(C_list) * length(sigma_list), 3);  
row = 1;
min_error = Inf;

for i = 1:length(C_list)
    for j = 1:length(sigma_list)
        C_val = C_list(i);
        sigma_val = sigma_list(j);
        model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);
        err_val = mean(double(predictions ~= yval));

        if err_val < min_error
            min_error = err_val;
            C = C_val;
            sigma = sigma_val;
        end
        results(row,:) = [C sigma min_error];
        row = row + 1;
    end
end

% disp(results)
% disp(C)
% disp(sigma)


% visualizeBoundary(X, y, model);


% =========================================================================

end