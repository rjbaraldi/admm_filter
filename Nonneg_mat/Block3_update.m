function X = Block3_update(Y,Z, Lambda, rho)


[N,Q]=size(Lambda); 
K = size(Y,1); 

    function [f,g] = froNorm(x,z, y, l, r)
        x = reshape(x, [N,K]); 
        f = r/2*norm(z - x*y+l/r, 'fro')^2;
        g = r*(z-x*y+l/r)*-y';
        g = g(:); 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 1000;
options.MaxIter = Q*(N+K); 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
funObj = @(X)froNorm(X, Z, Y, Lambda, rho) ;
LB = zeros(N*K,1);
UB = inf(N*K,1);

x = minConf_TMP(funObj, zeros(N*K,1), LB, UB, options); 
X = reshape(x, [N,K]); 


end