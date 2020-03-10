function nmf_test()
%note as of 3/6 we are still at pseudo-code levels
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
admm_simp.rho =1.1; %1.1

%subproblem routine options (for now these will be with CVX since they are
%convex)
admm_simp.alpha = norm(W,'fro');
admm_simp.beta = 1; 
admm_simp.augLag_stop = 1e-4; %L_rho^(k+1) - L_rho^k <= augLag_stop
admm_simp.dataM_stop = 1e-4; %||M - XY||_F/||M||_F <= dataM_stop for M data matrix
admm_simp.dims =[N, Q, R K]; 


[f3, X3,Y3] = admm_simple_3block(M, admm_simp); 


[f2, X2,Y2] = admm_simple_2block(M, admm_simp); 

fprintf('%s    2-Block: %1.4e      3-Block: %1.4e\n',...
    '||M - XY||_F',f2, f3);
fprintf('%s    2-Block: %1.4e      3-Block: %1.4e\n',...
    '||WH - XY||_F',norm(W*H - X2*Y2,'fro'), norm(W*H - X3*Y3,'fro'));


% fprintf('%s    ||W-X2||_F: %1.4e     ||Y2-H||_F: %1.4e\n',...
%     'Coord-Diff - 2vTrue',norm(X2-W,'fro'), norm(Y2-H,'fro'));
% 
% fprintf('%s    ||W-X3||_F: %1.4e     ||Y3-H||_F: %1.4e\n',...
%     'Coord-Diff - 3vTrue',norm(X3-W,'fro'), norm(Y3-H,'fro'));

fprintf('%s    ||X2-X3||_F: %1.4e     ||Y2-Y3||_F: %1.4e\n',...
    'Coord-Diff - 2v3',norm(X2-X3,'fro'), norm(Y2-Y3,'fro'));
fprintf('%s    %1.4e\n',...
    '||(XY) - (XY)||_F',norm(X2*Y2 - X3*Y3, 'fro')); 

    
end


