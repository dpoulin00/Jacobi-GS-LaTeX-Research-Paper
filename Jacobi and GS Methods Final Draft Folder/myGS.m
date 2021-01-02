function [conv,xnew,i,time] = myGS(A,b,xGuess,tol,itMax)
%This function implements the Gauss-Seidel method.
%A and b are the knowns in Ax = b.
%xGuess is our starting guess. If nothing is known about x, use the zero
%vector
%tol is the desired upper bound for the relative residual
%itMax is the maximum number of iterations our algorithm will go through

%First, we initialize L, U, D, and xold (our starting guess for x)
L = tril(A,-1);
U = A - tril(A);
D = diag(diag(A));
xold = xGuess;

%To save some calculations during the for loop, we calculate
%the norm of b and the inverse of L + D.
bNorm = norm(b);
LplusDInverse = inv(L+D);

%Here, we implement the Gauss-Seidel method itself
tic;
for i=1:itMax
    xnew = LplusDInverse*(b-U*xold);
    xold=xnew;
    relativeResidual = norm(b-A*xnew,2)/bNorm;
    %We check if the desired relative residual has been reached. If so, we
    %return. Otherwise, we continue
    if  relativeResidual <= tol
        conv = 1;
        time = toc;
        return
    end
end
%If we reach the max number of iterations, we return what we have so far as
%well as the current relative residual.
conv = 0;
time = toc;
return
end


