function admm_simple(M, options)

%feed in some options
rho = options.rho; 
alpha = options.alpha; 
beta = options.beta; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop; 

[N, Q, R, K] = options.dims; 

Z = randn([N,Q]); 
X = randn([N,K]); 
Lambda = rand([N,Q]); 

L_kp1 = augLag(Z, M, X, Y, Lambda); 
L_k = 0; 


%using CVX for now as a subproblem
while(L_kp1 - L_k >= augLag_stop && norm(M - X*Y,'fro')\M_fro<M_stop)
    
    Y = Yplus(Z, X, Lambda,rho); 
    
    
    
end




end



function Yplus(Z, X, Lambda, rho)

N, K = size(X); 
% x = X(:);
% z = z(:); 
% lambda = Lambda(:); 
% Id = eye(K); 
% Id = vec(:); 


cvx_begin quiet
    variable Y(N,K)
    minimize(rho/2*sum(sum_square_abs(Z - X*Y + Lambda/rho)))
    subject to
        Y(:)>=0
  
cvx_end
    
    
    
end