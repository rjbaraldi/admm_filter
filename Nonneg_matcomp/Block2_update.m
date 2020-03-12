function [X, Z] = Block2_update(Y,W, Lambda, rho, X_init, Z_init)


[N,Q]=size(Lambda); 
K = size(Y,1); 

    function [f,g] = froNorm(xz, y,w, l, r)
        z = reshape(xz(1:N*Q), [N,Q]); 
        x = reshape(xz(N*Q+1:end), [N,K]);
        f = r/2*norm(z - x*y+l/r, 'fro')^2+ .5*norm(z-w, 'fro')^2;
        gz = r*(z-x*y +l/r) + (z-w);
        gx = r*(z-x*y+l/r)*(-y');
        g = [gz(:); gx(:)]; 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 100;
options.MaxIter = Q*(N+K); 
options.Method = 'lbfgs';
options.optTol = 1e-3; 
% options.numDiff=1; 
funObj = @(XZ)froNorm(XZ, Y,W, Lambda, rho) ;
LB = [-inf(N*Q,1); zeros(N*K,1)]; 
UB = inf(Q*(N+K),1);

IC = [Z_init(:); X_init(:)]; 
xz = minConf_TMP(funObj, IC, LB, UB, options); 
Z = reshape(xz(1:N*Q), [N,Q]); 
X = reshape(xz(N*Q+1:end), [N,K]); 

end

