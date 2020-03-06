function nmf_test()
%note as of 3/6 we are still at pseudo-code levels

%Generate Data (using section 4.2 conditions)
N = 5000; %data row dimension
Q = N; %data column dimension
R = 1000; %real data rank cutoff
K = 200; %low-rank cutoff

%M = WH + N
%   where W is \R^{NxR}, H = \R^{RxQ}, cN =\R^{NxQ} but uniform(0,1) iid
%   noise with sigma = 0.01^2
W = rand(N,R); H = rand(R,Q); cN = randn([N,Q])*.01^2; 
M = W*H + cN; 



%initialize things in section 4.2 of NMFbilinearADMM.pdf
admm_simp.rho = 1.1;

%subproblem routine options (for now these will be with CVX since they are
%convex)
admm_simp.alpha = norm(W,'fro');
admm_simp.beta = 1; 
admm_simp.augLag_stop = 1e-4; %L_rho^(k+1) - L_rho^k <= augLag_stop
admm_simp.dataM_stop = 1e-4; %||M - XY||_F/||M||_F <= dataM_stop for M data matrix
admm_simp.dims =[N, Q, R K]; 


[f, X,Y] = admm_simple(M, admm_simp); 


end


