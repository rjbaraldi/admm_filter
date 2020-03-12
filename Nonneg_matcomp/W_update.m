function W = W_update(Z, M, Po, W_init)

[N,Q] = size(W_init);  

    function [f,g] = froNorm(w,z)
        w = reshape(w, [N,Q]); 
        f = .5*norm(z - w, 'fro')^2;
        g = -(z - w);
        g = g(:); 
    end
    function w = projBound(w, m, Po) 
        w(Po) = m(Po);   
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 1000;
options.MaxIter = N*Q; 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
% options.numDiff=1; 
funObj = @(W)froNorm(W,Z) ;
Proj = @(W)projBound(W, M(:), Po); 

w = minConf_SPG(funObj, W_init(:), Proj, options); 
W = reshape(w, [N, Q]); 
    
    
end
