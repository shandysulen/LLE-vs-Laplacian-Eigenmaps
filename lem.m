function [Y] = lem(X,K,sigma2)

X = transpose(X);
m = size(X, 1);
dt = squareform(pdist(X));
[srtdDt, srtdIdx] = sort(dt, 'ascend');
dt = srtdDt(1:K+1,:);
nidx = srtdIdx(1:K+1,:);

% Compute weights
tempW = exp(-dt.^2/sigma2);

% Build weight matrix
i = repmat(1:m, K+1, 1);
w = sparse(i(:), double(nidx(:)), tempW(:),m,m);
W = max(W,W');

% Create Normalized Graph Laplacian
ld = diag(sum(W,2).^(-1/2));
DO = ld*W(ld;
DO = max(DO,DO');

% Get eigenvectors & output Y
[v,d] = eigs(DO,10,'la');
Y = [transpose(v(:,2)); transpose(v(:,4))];