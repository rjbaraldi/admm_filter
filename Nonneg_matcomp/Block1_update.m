function [Y,W] = Block1_update(Z, X, Lambda, rho,M, Y_init, W_init, P_omega)

[K,Q] = size(Y_init); 
[N,Q] = size(W_init); 

    function [f,g] = froNorm(yw, x, z, l, r)
        y = reshape(yw(1:K*Q), [K,Q]); 
        w = reshape(yw(K*Q+1:end), [N,Q]); 
        f = .5*norm(z - w,'fro') + r/2*norm(z - x*reshape(y,[K,Q])+l/r, 'fro')^2;
        gy = -r*x'*(z - x*reshape(y, [K,Q]) + l/r);
        gw = -(z-w); 
        g = [gy(:); gw(:)]; 
    end
    
    function p = projBound(yw, m, Po)
        y = yw(1:K*Q); 
        w = yw(K*Q+1:end);
        y(y<0) = 0; 
        w(Po) = m(Po); 
        p = [y;w]; 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 100;
options.MaxIter = K*Q; 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
% options.numDiff=1; 
funObj = @(YW)froNorm(YW, X, Z, Lambda, rho) ;
Proj = @(YW)projBound(YW, M(:), P_omega); 

yw = minConf_SPG(funObj, [Y_init(:);W_init(:)], Proj, options); 
Y = reshape(yw(1:K*Q), [K, Q]); 
W = reshape(yw(K*Q+1:end), [N,Q]); 
    
    
end
