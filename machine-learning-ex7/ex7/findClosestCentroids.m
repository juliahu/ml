function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
% Initialize with
% matrix of m x K, 300 X 3
distance = zeros(size(X,1), K);
%disp(X(1:3, :));

% iterate over centroids
for i = 1:K
    % diff of x and centroid
    A = bsxfun(@minus, X, centroids(i,:));

    % square the diff
    B = bsxfun(@power, A, 2);

    % sum the above by row, assign by column vector to distance 
    distance(:,i) = sum(B, 2); 
end
%disp(distance(1:3, :));

for j = 1:length(idx)
    % return idx(m x 1) as the vector of the indexes of the locations w/min distance
    [val,index] = min(distance(j,:), [], 2);
    idx(j) = index;
end
% =============================================================

end

