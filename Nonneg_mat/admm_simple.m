function [f, X, Y] = admm_simple(M, options)

%feed in some options
rho = options.rho; 
alpha = options.alpha; 
beta = options.beta; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop; 

N = options.dims(1);
Q = options.dims(2); 
R = options.dims(3); 
K = options.dims(4); 

Z = randn([N,Q]); 
X = randn([N,K]); 
Y = randn([K,Q]); 
Lambda = rand([N,Q]); 

L_kp1 = augLag(X, Y, Z, M, Lambda, rho);  
L_iter = Inf; 
M_fro = norm(M, 'fro'); 
Mf_iter = norm(M - X*Y, 'fro'); 
iter = 1; 

fprintf('%s  %s  %s   %s   %s  %s  %s\n',...
    'Iter', 'Lp - Lp+', '||M - XY||_F/||M||_F', '||Y+ - Y||_F', '||X+ - X||_F', '||Z+ - Z||_F', '||Lambda+ - Lambda||_F'); 
fprintf('----------------------------------------------------------------------------------------------------------\n');
    
%using CVX for now as a subproblem
while(L_iter >= augLag_stop && Mf_iter\M_fro>M_stop)
    L_k = L_kp1; 
    Yp= Block1_update(Z, X, Lambda, rho, Y); 
    [Xp, Zp] = Block2_update(Yp,Lambda, rho,M, X,Z);
    
%Uncomment below for 3-block admm
%     Xp = Block2_update(Yp,Z, Lambda, rho); 
%     Zp = (M-Lambda+rho*Xp*Yp)/(1+rho); 
    
    
    
    Lambdap = Lambda + (Zp - Xp*Yp)*rho;
    
    
    L_kp1 = augLag(Xp, Yp, Zp, M, Lambdap, rho);
    L_iter = abs(L_k - L_kp1); 
    Mf_iter = norm(M - Xp*Yp,'fro');
    
    fprintf('%i    %1.4e      %1.4e         %1.4e     %1.4e    %1.4e      %1.4e\n',...
    iter, L_iter, Mf_iter/M_fro, norm(Yp-Y,'fro'), norm(Xp-X,'fro'), norm(Zp-Z,'fro'), norm(Lambdap-Lambda,'fro'));
    
    
    
    X = Xp; 
    Y = Yp; 
    Z = Zp; 
    Lambda = Lambdap; 
    iter = iter+1; 
    
     

end

f = norm(X*Y - M, 'fro')^2; 


end



function L = augLag(X, Y, Z, M, Lambda, rho)

L = .5*norm(Z - M, 'fro')^2 + trace(Lambda'*(Z - X*Y)) + rho/2*norm(Z - X*Y, 'fro')^2; 

end
