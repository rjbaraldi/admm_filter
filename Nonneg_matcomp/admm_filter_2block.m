function [f, X, Y] = admm_filter_2block(M, options)

%feed in some options
rho = options.rho; 
alpha = options.alpha; 
beta = options.beta; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop; 
P_omega = options.P_omega; 

N = options.dims(1);
Q = options.dims(2); 
R = options.dims(3); 
K = options.dims(4); 

Z = randn([N,Q]); 
X = randn([N,K]); 
Y = randn([K,Q]); 
W = randn([N,Q]); 
Lambda = randn([N,Q]); 

L_kp1 = augLag(X, Y, Z, M, Lambda, rho);  
L_iter = Inf; 
W_fro = norm(W, 'fro'); 
Wf_iter = norm(W - X*Y, 'fro'); 
iter = 1; 


fprintf('%s  %s  %s   %s   %s  %s  %s %s\n',...
    'Iter', 'Lp - Lp+', '||M - XY||_F/||M||_F', '||Y+ - Y||_F', '||X+ - X||_F', '||Z+ - Z||_F', '||Lambda+ - Lambda||_F', 'rho-status'); 
fprintf('--------------------------------------------------------------------------------------------------------------------------\n');
    

while(L_iter >= augLag_stop && Wf_iter\W_fro>M_stop) %keeping the same 'tolerance' constraints as in the paper
    L_k = L_kp1; 
    
    
    %solve aug-Lag for two steps - such that sufficient reduction holds
    while omega_p>omega && eta_p > eta
    [Yp, Wp] = Block1_update(Z, X, Lambda, rho,M, Y, W, P_omega);  
    [Xp, Zp] = Block2_update(Yp,Wp, Lambda, rho, X,Z); 
    
    
    end
    
    if norm(Zp - Xp*Yp, 'fro')^2<norm(Z - X*Y,'fro')^2
        Lambdap = Lambda + (Zp - Xp*Yp)*rho;
    else
        rho = 2*rho; 
    end
    
    
%%%%%%%%%Note - below is the simple filter method currently in the paper
%%%%%%%%%draft. 
%     %solve aug-Lag for two steps - such that sufficient reduction holds
%     %(input sufficient reduction - ||Z - XY||_F^2 in the notes)
%     [Yp, Wp] = Block1_update(Z, X, Lambda, rho,M, Y, W, P_omega);  
%     [Xp, Zp] = Block2_update(Yp,Wp, Lambda, rho, X,Z); 
%     
%     
%     if norm(Zp - Xp*Yp, 'fro')^2<norm(Z - X*Y,'fro')^2
%         Lambdap = Lambda + (Zp - Xp*Yp)*rho;
%     else
%         rho = 2*rho; 
%     end
    
    
    L_kp1 = augLag(Xp, Yp, Zp, M, Lambdap, rho);
    L_iter = abs(L_k - L_kp1); 
    Wf_iter = norm(Wp - Xp*Yp,'fro');
    W_fro = norm(Wp, 'fro');
    
    fprintf('%i    %1.4e      %1.4e         %1.4e     %1.4e    %1.4e      %1.4e          %1.4e\n',...
    iter, L_iter, Wf_iter/W_fro, norm(Yp-Y,'fro'), norm(Xp-X,'fro'), norm(Zp-Z,'fro'), norm(Lambdap-Lambda,'fro'), rho);
    
    
    
    X = Xp; 
    Y = Yp; 
    Z = Zp; 
    Lambda = Lambdap; 
    iter = iter+1; 
    
     

end

f = norm(X*Y - M, 'fro')^2; 


end

function eta_comp()
%compute eta = ||c(x)|| => ||Z - XY|| = 0 and Po(W - M)=0? 

end


function omega_comp()
%compute omega = ||min{x - l, max{x-u, \nabla_x Lp(x,y)}}||


end

