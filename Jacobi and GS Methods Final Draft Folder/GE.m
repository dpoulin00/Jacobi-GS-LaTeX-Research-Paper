function [Aref, x, time] = GE(A,b)
%
% [Aref, x, time] = GE(A,b)
%
% GE is the basic Gaussian Elimination algorithm (no pivoting) and
% backsubstitution for solving the linear system Ax = b. 
%
% The inputs are as follows:
%
% A = the square matrix in Ax = b
% b = the data vector in Ax = b
%
% The outputs are as follows:
%
% Aref = row echelon form of matrix A
% x = computed solution vector to Ax = b
% time = algorithm run time (in seconds) 
[nrow, ncol] = size(A);
[mrow, mcol] = size(b);
C = [A b]; % define augmented matrix
% Check that Ax = b is dimensionally possible
if nrow~=ncol
    error('A must be square')
end
if nrow~=mrow
    error('A and b must have the same number of rows')
end
m = zeros(nrow-1,nrow-1); % place holder for multipliers
tic % start timer
% Begin GE
for k = 1:nrow-1
    if C(k,k) == 0
        error(['divisor too small, cannot continue'])
    end
    % compute multiplier vector
    m(k+1:nrow,k) = -C(k+1:nrow,k)/C(k,k);
    % perform row ops below current row
    C(k+1:nrow,:) = C(k+1:nrow,:)+m(k+1:nrow,k)*C(k,:); 
end 
if C(nrow,nrow) == 0
    error('A is singular')
end
% backsolve for x
x(nrow,1) = C(nrow,nrow+1)/C(nrow,nrow);
for k = nrow-1:-1:1
    x(k,1) = (C(k,nrow+1)-C(k,k+1:nrow)*x(k+1:nrow,1))/C(k,k);
end
time = toc; % end timer
Aref = C(:,1:nrow);