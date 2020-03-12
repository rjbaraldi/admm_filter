function X = X_update(Y,Z, Lambda, rho, X_init)

[N,K] = size(X_init);  

    function [f,g] = froNorm(x, y, z, l, r)
        x = reshape(x, [N,K]); 
        f = r/2*norm(z - x*y+l/r, 'fro')^2;
        g = -r*(z - x*y + l/r)*y'; %check maybe? 
        g = g(:); 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 1000;
options.MaxIter = N*K; 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
% options.numDiff=1; 
funObj = @(X)froNorm(X,Y, Z, Lambda, rho) ;
LB = zeros(K*N,1); 
UB = inf(K*N,1); 

x = minConf_TMP(funObj, X_init(:), LB, UB, options); 
X = reshape(x, [N,K]); 
    
    
end
