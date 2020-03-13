function [xk, omega, eta] = admm_filter_2block(M, options, funcB1, ProjB1, funcB2, ProjB2, stop_crit)

%feed in some options
rho = options.rho; 
% alpha = options.alpha; 
% beta = options.beta; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop; 

N = options.dims(1);
Q = options.dims(2); 
% R = options.dims(3); 
K = options.dims(4); 

% xk = [X(:); Y(:); Z(:); W(:)];
xk = options.x; 
Lambda = randn([N,Q]); 



%put in options for sorting variables
Lagopts.ind.X = options.ind.X; 
Lagopts.ind.Y = options.ind.Y;
Lagopts.ind.Z = options.ind.Z;
Lagopts.ind.W = options.ind.W; 
Lagopts.dims = options.dims;
Lagopts.rho = rho; 
Lagopts.Po = options.P_omega; 
Lagopts.M = M; 
Lagopts.LB = options.LB; 
Lagopts.UB = options.UB; 


solver_opts.verbose=0;
solver_opts.display = 'iter'; 
solver_opts.maxFunEvals=100; 
solver_opts.MaxIter = 100; %can change based on matrix size 
solver_opts.Method = 'lbfgs'; 
solver_opts.optTol = 1e-3; 
solver_opts.numDiff = 0; %1 does FD differentiation, 2 does complex fd


[L_kp1,~] = augLag(xk, Lambda, Lagopts);  
L_iter = Inf; 
[OptTol, ~] = stop_crit(xk, Lambda, Lagopts); 
k = 1; 


fprintf('%s  %s  %s   %s   %s  %s \n',...
    'Iter', 'Lp - Lp+', '||M - XY||_F/||M||_F', '||x+ - x||_F', '||Lambda+ - Lambda||_F', 'rho-status'); 
fprintf('------------------------------------------------------------------------------------------------\n');
    
omega = omega_comp(xk,Lambda, Lagopts); 
eta = eta_comp(xk, Lagopts); 

while(L_iter >= augLag_stop && OptTol>M_stop) %keeping the same 'tolerance' constraints as in the paper
    L_k = L_kp1; 
    
    %solve aug-Lag for two steps - such that sufficient reduction holds
    omega_p = -Inf; 
    eta_p = -Inf; 
    j = 0; 
    while (omega_p > omega(k) && eta_p > eta(k)) || j==0 %j loop - cancels after one iteration right now
        xj = minConf_SPG(@(b1)funcB1(b1, Lambda, Lagopts), xk, @(x)ProjB1(x,Lagopts), solver_opts);
        xj = minConf_SPG(@(b2)funcB2(b2, Lambda, Lagopts), xj, @(x)ProjB2(x,Lagopts), solver_opts);  
        
        
        omega = [omega, omega_comp(xj,Lambda, Lagopts)]; 
        eta = [eta, eta_comp(xj, Lagopts)];
        j = j+1; 
    
    end
    xk = xj; 
%     if norm(Zp - Xp*Yp, 'fro')^2<norm(Z - X*Y,'fro')^2
%         Lambdap = Lambda + (Zp - Xp*Yp)*rho;
%     else
%         rho = 2*rho; 
%     end
    
    
%%%%%%%%%Note - below is the simple filter method currently in the paper
%%%%%%%%%draft.
%     
%     
%     if norm(Zp - Xp*Yp, 'fro')^2<norm(Z - X*Y,'fro')^2

%     else
%         rho = 2*rho; 
%     end
    
    
    [L_kp1, ~] = augLag(xk, Lambda, Lagopts);
    L_iter = abs(L_k - L_kp1); 
    [OptTol, Lambdap] = stop_crit(xk, Lambda, Lagopts);
    
%     fprintf('%i    %1.4e      %1.4e         %1.4e     %1.4e    %1.4e      %1.4e          %1.4e\n',...
%     iter, L_iter, Wf_iter/W_fro, norm(Yp-Y,'fro'), norm(Xp-X,'fro'), norm(Zp-Z,'fro'), norm(Lambdap-Lambda,'fro'), rho);
%     
 fprintf('%i    %1.4e      %1.4e         %1.4e       %1.4e          %1.4e  \n',...
    k, L_iter, OptTol, norm(xk - xj,'fro'), norm(Lambdap-Lambda,'fro'), rho);
       
   
    Lambda = Lambdap; 
    k = k+1; 
    
     

end




end

function eta = eta_comp(x, Lagopts)
%compute eta = ||c(x)|| => ||Z - XY|| = 0 and Po(W - M)=0? 
        N = Lagopts.dims(1);
        Q = Lagopts.dims(2); 
        % R = options.dims(3); 
        K = Lagopts.dims(4);
        
        X = reshape(x(Lagopts.ind.X(1):Lagopts.ind.X(2)), [N,K]);
        Y = reshape(x(Lagopts.ind.Y(1):Lagopts.ind.Y(2)), [K,Q]); 
        Z = reshape(x(Lagopts.ind.Z(1):Lagopts.ind.Z(2)), [N,Q]);
        eta = norm(Z - X*Y, 'fro'); 
        

end


function omega = omega_comp(x, Lambda, Lagopts)
%compute omega = ||min{x - l, max{x-u, \nabla_x Lp(x,y)}}||
    l = Lagopts.LB; 
    u = Lagopts.UB; 
    [~, L_grad] = augLag(x,Lambda, Lagopts);
    omega = min(x-l, max(x-u, L_grad)); 
    omega = norm(omega); 

end

