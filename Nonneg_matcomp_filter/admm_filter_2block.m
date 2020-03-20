function [xk, Filter] = admm_filter_2block(M, options, funcB1, ProjB1, funcB2, ProjB2, stop_crit, lnSrch)

%feed in some options
rho = options.rho; 
% alpha = options.alpha; 
% beta = options.beta; 
IterMax = options.IterMax; 
augLag_stop = options.augLag_stop; 
M_stop = options.dataM_stop; 

N = options.dims(1);
Q = options.dims(2); 
% R = options.dims(3); 
K = options.dims(4);
gamma = options.gamma; 
beta = options.beta; 

% xk = [X(:); Y(:); Z(:); W(:)];
xk = options.x; 
Lambdak = randn([N,Q]); 



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


[L_kp1,~] = augLag(xk, Lambdak, Lagopts);  
L_iter = Inf; 
[OptTol, ~] = stop_crit(xk, Lambdak, Lagopts); 
k = 1; 


fprintf('%s  %s  %s   %s   %s  %s \n',...
    'Iter', 'Lp - Lp+', '||M - XY||_F/||M||_F', '||x+ - x||_F', '||Lambda+ - Lambda||_F', 'rho-status'); 
fprintf('------------------------------------------------------------------------------------------------\n');
    
Filter = [eta_comp(xk, Lagopts);  omega_comp(xk,Lambdak, Lagopts)]; 

while(L_iter >= augLag_stop && OptTol>M_stop && k<IterMax) %keeping the same 'tolerance' constraints as in the paper
    L_k = L_kp1; 
    
    %solve aug-Lag for two steps - such that sufficient reduction holds
    j = 0; 
    xj = xk; 
    Lambdaj = Lambdak; 
    omegaj = omega_comp(xj,Lambdak, Lagopts); 
    etaj = eta_comp(xj, Lagopts); 
    UBD = max(min(Filter(2, :))/gamma, beta*min(Filter(1,:))); %not directly enforcing eta>0 but should be implicit I think? 
    while (omegaj > Filter(2,end) && etaj > Filter(1,end)) || j==0 %j loop - cancels after one iteration right now
        %block 1
        xj = minConf_SPG(@(b1)funcB1(b1, Lambdaj, Lagopts), xj, @(x)ProjB1(x,Lagopts), solver_opts);
        %block 2
        xj = minConf_SPG(@(b2)funcB2(b2, Lambdaj, Lagopts), xj, @(x)ProjB2(x,Lagopts), solver_opts);  
        
        
        
        %compute filter points for xj
        etaj = eta_comp(xj, Lagopts); 
%         omegaj = omega_comp(xj, Lambda, Lagopts); 
        if etaj>UBD
            %insert linesearch to find acceptable (etaj, omegaj) 
            Lagopts.rho = 2*Lagopts.rho; %increase rho before doing conditions? 
            [xj, alpha] = lnSrch(xj, UBD, Lagopts); %changed it to an eta search 
            etaj = eta_comp(xj, Lagopts);
            omegaj = omega_comp(xj, Lambdaj, Lagopts);
            
            fprintf('   etaj = %1.4e      UBD=%1.4e\n', etaj, UBD);
            
        end
        etaj = eta_comp(xj, Lagopts); 
        omegaj = omega_comp(xj, Lambdaj, Lagopts); 
        [OptTol, Lambdaj] = stop_crit(xj, Lambdaj, Lagopts); 
        j = j+1; 
    
    end

    
    
    
    [L_kp1, ~] = augLag(xj, Lambdak, Lagopts);
    L_iter = abs(L_k - L_kp1); 
    
  
 fprintf('%i    %1.4e      %1.4e         %1.4e       %1.4e          %1.4e  \n',...
    k, L_iter, OptTol, norm(xk - xj,'fro'), norm(Lambdak-Lambdaj,'fro'), Lagopts.rho);
       


  
    %update etamin, omegamin
    if etaj>0
        Filter = [Filter, [etaj; omegaj]]; 
    end
    xk = xj; 
    Lambdak = Lambdaj; 
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
        eta = norm(Z - X*Y, 'fro')^2; 
        

end


function omega = omega_comp(x, Lambda, Lagopts)
%compute omega = ||min{x - l, max{x-u, \nabla_x Lp(x,y)}}||
    l = Lagopts.LB; 
    u = Lagopts.UB; 
    [~, L_grad] = augLag(x,Lambda, Lagopts);
    omega = min(x-l, max(x-u, L_grad)); 
    omega = norm(omega); 

end

