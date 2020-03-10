function Y = Block1_update(Z, X, Lambda, rho, Y_init)

[K,Q] = size(Y_init);  

    function [f,g] = froNorm(y, x, z, l, r)
        f = r/2*norm(z - x*reshape(y,[K,Q])+l/r, 'fro')^2;
        g = -r*x'*(z - x*reshape(y, [K,Q]) + l/r);
        g = g(:); 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 1000;
options.MaxIter = K*Q; 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
% options.numDiff=1; 
funObj = @(Y)froNorm(Y, X, Z, Lambda, rho) ;
LB = zeros(K*Q,1); 
UB = inf(K*Q,1); 

y = minConf_TMP(funObj, Y_init(:), LB, UB, options); 
Y = reshape(y, [K, Q]); 
    
    
end
