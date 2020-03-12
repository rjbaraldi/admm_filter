function nmfc_filter_test()
%note as of 3/11 we are at decent running levels
rng(5);
%Generate Data (using section 4.2 conditions)
N = 50; %data row dimension (5000 in paper)
Q = N; %data column dimension
R = Q/10; %real data rank cutoff (1000 in paper)
K = Q/2; %low-rank cutoff (200 in paper)

%M = WH + N
%   where W is \R^{NxR}, H = \R^{RxQ}, cN =\R^{NxQ} but uniform(0,1) iid
%   noise with sigma = 0.01^2
W = rand(N,R); H = rand(R,Q); cN = randn([N,Q])*.01^2; 
M = W*H + cN; 



%initialize things in section 4.2 of NMFbilinearADMM.pdf
admm_simp.rho =.1; %1.1

%subproblem routine options (for now these will be with CVX since they are
%convex)
admm_simp.alpha = norm(W,'fro');
admm_simp.beta = 1; 
admm_simp.augLag_stop = 1e-4; %L_rho^(k+1) - L_rho^k <= augLag_stop
admm_simp.dataM_stop = 1e-4; %||M - XY||_F/||M||_F <= dataM_stop for M data matrix
admm_simp.dims =[N, Q, R K];


p_miss = .25;
num_miss = floor(p_miss*numel(M)); 
ind_miss = randperm(numel(M), num_miss)';
% P_omega = speye(numel(M));
% P_omega(ind_miss,:)=[]; 
admm_simp.P_omega = ind_miss; %should sample all the observed values

 


[f2, X2,Y2] = admm_filter_2block(M, admm_simp); 

fprintf('%s    2-Block: %1.4e  \n',...
    '||M - XY||_F',f2);
fprintf('%s    2-Block: %1.4e  \n',...
    '||WH - XY||_F',norm(W*H - X2*Y2,'fro'));

 

    
end


