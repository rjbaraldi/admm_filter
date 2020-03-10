function [X, Z] = Block2_update(Y, Lambda, rho, X_init, Z_init)


[N,Q]=size(Lambda); 
K = size(Y,1); 

    function [f,g] = froNorm(xz, y, l, r)
        z = reshape(xz(1:N*Q), [N,Q]); 
        x = reshape(xz(N*Q+1:end), [N,K]);
        Id = eye(size(z,2)); 
        f = r/2*norm(z - x*y+l/r, 'fro')^2;
        g = r*(z-x*y+l/r)*[Id, -y'];
        g = g(:); 
    end
 
options = [];
options.verbose=0;
options.display = 'iter';
options.maxFunEvals = 1000;
options.MaxIter = Q*(N+K); 
options.Method = 'lbfgs';
options.optTol = 1e-6; 
funObj = @(XZ)froNorm(XZ, Y, Lambda, rho) ;
LB = [-inf(N*Q,1); zeros(N*K,1)]; 
UB = inf(Q*(N+K),1);

IC = [Z_init, X_init]; 
xz = minConf_TMP(funObj, IC(:), LB, UB, options); 
X = reshape(xz(1:N*K), [N,K]); 
Z = reshape(xz(N*K+1:end), [N,Q]); 

end

% function X = Block2_update(Y,Z, Lambda, rho)
% 
% 
% [N,Q]=size(Lambda); 
% K = size(Y,1); 
% 
%     function [f,g] = froNorm(x,z, y, l, r)
%         x = reshape(x, [N,K]); 
%         f = r/2*norm(z - x*y+l/r, 'fro')^2;
%         g = r*(z-x*y+l/r)*-y';
%         g = g(:); 
%     end
%  
% options = [];
% options.verbose=0;
% options.display = 'iter';
% options.maxFunEvals = 1000;
% options.MaxIter = Q*(N+K); 
% options.Method = 'lbfgs';
% options.optTol = 1e-6; 
% funObj = @(X)froNorm(X, Z, Y, Lambda, rho) ;
% LB = zeros(N*K,1);
% UB = inf(N*K,1);
% 
% x = minConf_TMP(funObj, zeros(N*K,1), LB, UB, options); 
% X = reshape(x, [N,K]); 
% 
% 
% end

% 
% cvx_begin quiet
%     variable X(N,K)
%     minimize(rho/2*sum(sum_square_abs(Z - X*Y + Lambda/rho)))
%     subject to
%         0<=X(:)
%   
% cvx_end
% cvx_clear