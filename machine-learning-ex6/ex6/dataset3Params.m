function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

cSigma = [0.01 0.03 0.1 0.3 1 3 10 30];
len = length(cSigma);
error = 1;

% all possible pairs of C and sigma
for c = 1:len
    for s = 1:len

       % train the model using svmTrain
       model = svmTrain(X, y, cSigma(c), @(x1, x2) gaussianKernel(x1, x2, cSigma(s)));

       % compute predictions w/svmPredict
       pred = svmPredict(model, Xval);

       % get the error btw predictions and yval
       predictionError = mean(double(pred ~= yval));
 
       % is new min error? save C and sigma if it is
       if (predictionError < error)
           error = predictionError;
           C = cSigma(c);
           sigma = cSigma(s);
       end 
    end
end


% =========================================================================

end
