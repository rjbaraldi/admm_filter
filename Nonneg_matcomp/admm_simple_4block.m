function [f, X, Y] = admm_simple_4block(M, options)

%feed in some options
rho = options.rho; 
alpha = options.alpha; 
beta = options.beta; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop;
Po = options.P_omega; 

N = options.dims(1);
Q = options.dims(2); 
R = options.dims(3); 
K = options.dims(4); 

Z = randn([N,Q]); 
X = randn([N,K]); 
Y = randn([K,Q]);
W = randn([N,Q]); 
Lambda = rand([N,Q]); 

L_kp1 = augLag(X, Y, Z, W, Lambda, rho);  
L_iter = Inf; 
W_fro = norm(M, 'fro'); 
Wf_iter = norm(W - X*Y, 'fro'); 
iter = 1; 

fprintf('%s  %s  %s   %s   %s  %s  %s\n',...
    'Iter', 'Lp - Lp+', '||W - XY||_F/||W||_F', '||Y+ - Y||_F', '||X+ - X||_F', '||Z+ - Z||_F', '||Lambda+ - Lambda||_F'); 
fprintf('----------------------------------------------------------------------------------------------------------\n');
    
%using CVX for now as a subproblem
while(L_iter >= augLag_stop && Wf_iter\W_fro>M_stop)
    L_k = L_kp1; 
    Yp= Y_update(Z, X, Lambda, rho, Y); 
    Wp = W_update(Z,M, Po, W); 
    Zp = (Wp-Lambda+rho*X*Yp)/(1+rho); 
    Xp = X_update(Yp,Zp, Lambda, rho, X); 
    
    
    
    
    
    Lambdap = Lambda + (Zp - Xp*Yp)*rho;
    
    
    L_kp1 = augLag(Xp, Yp, Zp, M, Lambdap, rho);
    L_iter = abs(L_k - L_kp1); 
    Wf_iter = norm(Wp - Xp*Yp,'fro');
    W_fro = norm(Wp, 'fro');
    
    fprintf('%i    %1.4e      %1.4e         %1.4e     %1.4e    %1.4e      %1.4e\n',...
    iter, L_iter, Wf_iter/W_fro, norm(Yp-Y,'fro'), norm(Xp-X,'fro'), norm(Zp-Z,'fro'), norm(Lambdap-Lambda,'fro'));
    
    
    
    X = Xp; 
    Y = Yp; 
    Z = Zp; 
    Lambda = Lambdap; 
    iter = iter+1; 
    
     

end

f = norm(X*Y - M, 'fro')^2; 


end

