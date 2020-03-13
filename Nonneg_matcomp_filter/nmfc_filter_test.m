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
admm_simp.rho =1.1; %1.1

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


Zi = randn([N,Q]); 
Xi = randn([N,K]); 
Yi = randn([K,Q]); 
Wi = randn([N,Q]); 
x = [Xi(:); Yi(:); Zi(:); Wi(:)];


%put in options for sorting variables
admm_simp.ind.X = [1,N*K]; 
admm_simp.ind.Y = [N*K+1,N*K+K*Q];
admm_simp.ind.Z = [K*(N+Q)+1,K*(N+Q)+N*Q];
admm_simp.ind.W = [K*(N+Q)+1+N*Q,K*(N+Q)+N*Q+N*Q]; 
admm_simp.M = M; 
admm_simp.x = x; 

admm_simp.LB = -inf(size(x)); admm_simp.LB(admm_simp.ind.X(1):admm_simp.ind.Y(2)) = 0; 
admm_simp.UB = inf(size(x)); 


[x, omega, eta] = admm_filter_2block(M, admm_simp, @YWupdate, @YWprojBound, @ZXupdate, @ZXprojBound, @stop_crit); 



Xf = reshape(x(admm_simp.ind.X(1):admm_simp.ind.X(2)), [N,K]); 
Yf = reshape(x(admm_simp.ind.Y(1):admm_simp.ind.Y(2)), [K,Q]); 
% Z = reshape(x(admm_simp.ind.Z(1):admm_simp.ind.Z(2)), [N,Q]); 
Wf = reshape(x(admm_simp.ind.W(1):admm_simp.ind.W(2)), [N,Q]);

fprintf('%s    2-Block: %1.4e  \n',...
    '||M - XY||_F',norm(M - Xf*Yf, 'fro'));
fprintf('%s    2-Block: %1.4e  \n',...
    '||WH - XY||_F',norm(W*H - Xf*Yf,'fro'));

fprintf('%s    2-Block: %1.4e  \n',...
    '||W - M||_F',norm(Wf - Xf*Yf,'fro'));

loglog(eta, omega, '*')
set(gca, 'Fontsize', 22, 'Fontweight', 'Bold')
xlabel('\eta')
ylabel('\omega')
    
end

function [f,g] = YWupdate(x, varargin)
        Lambda = varargin{1}; 
        Lagopts = varargin{2}; 
        N = Lagopts.dims(1);
        Q = Lagopts.dims(2); 
        % R = options.dims(3); 
        K = Lagopts.dims(4);
        rho = Lagopts.rho; 

        % [X(:); Y(:); Z(:); W(:)]
        xi = Lagopts.ind.X; 
        yi = Lagopts.ind.Y;
        zi = Lagopts.ind.Z; 
        wi = Lagopts.ind.W; 


        X = reshape(x(xi(1):xi(2)), [N,K]); 
        Y = reshape(x(yi(1):yi(2)), [K,Q]); 
        Z = reshape(x(zi(1):zi(2)), [N,Q]); 
        W = reshape(x(wi(1):wi(2)), [N,Q]); 
        
        f = .5*norm(Z - W,'fro') + rho/2*norm(Z - X*Y+Lambda/rho, 'fro')^2;
        gy = -rho*X'*(Z - X*Y + Lambda/rho);
        gw = -(Z-W); 
        %right now just keep 0-gradients wherever x,z are
        g = [zeros(numel(X),1); gy(:); zeros(numel(Z),1); gw(:)]; 
end

function p = YWprojBound(x, varargin)
        
        Lagopts = varargin{1}; 

        % [X(:); Y(:); Z(:); W(:)]
        xi = Lagopts.ind.X; 
        yi = Lagopts.ind.Y;
        zi = Lagopts.ind.Z; 
        wi = Lagopts.ind.W; 


        X = x(xi(1):xi(2)); 
        Y = x(yi(1):yi(2)); 
        Z = x(zi(1):zi(2)); 
        W = x(wi(1):wi(2)); 
        M = Lagopts.M;
        Po = Lagopts.Po; 
        
        Y(Y<0) = 0; 
        W(Po) = M(Po); 
        p = [X; Y; Z; W]; 
end


function [f,g] = ZXupdate(x, varargin)
        Lambda = varargin{1}; 
        Lagopts = varargin{2}; 
        N = Lagopts.dims(1);
        Q = Lagopts.dims(2); 
        % R = options.dims(3); 
        K = Lagopts.dims(4);
        rho = Lagopts.rho; 

        % [X(:); Y(:); Z(:); W(:)]
        xi = Lagopts.ind.X; 
        yi = Lagopts.ind.Y;
        zi = Lagopts.ind.Z; 
        wi = Lagopts.ind.W; 


        X = reshape(x(xi(1):xi(2)), [N,K]); 
        Y = reshape(x(yi(1):yi(2)), [K,Q]); 
        Z = reshape(x(zi(1):zi(2)), [N,Q]); 
        W = reshape(x(wi(1):wi(2)), [N,Q]); 
        

        f = rho/2*norm(Z - X*Y+Lambda/rho, 'fro')^2+ .5*norm(Z-W, 'fro')^2;
        nn = rho*(Z-X*Y +Lambda/rho);
        gz =  nn + (Z-W);
        gx = nn*(-Y');
        g = [gx(:);zeros(numel(Y),1); gz(:);zeros(numel(W),1)]; 
end


function p = ZXprojBound(x, varargin)
        
        Lagopts = varargin{1}; 

        % [X(:); Y(:); Z(:); W(:)]
        xi = Lagopts.ind.X; 
        yi = Lagopts.ind.Y;
        zi = Lagopts.ind.Z; 
        wi = Lagopts.ind.W; 


        X = x(xi(1):xi(2)); 
        Y = x(yi(1):yi(2)); 
        Z = x(zi(1):zi(2)); 
        W = x(wi(1):wi(2)); 
        X(X<0) = 0;  
        p = [X; Y; Z; W]; 
end

function [OptTol, Lambdap] = stop_crit(x, varargin)
        Lambda = varargin{1}; 
        Lagopts = varargin{2}; 
        N = Lagopts.dims(1);
        Q = Lagopts.dims(2); 
        % R = options.dims(3); 
        K = Lagopts.dims(4);
        rho = Lagopts.rho;

        X = reshape(x(Lagopts.ind.X(1):Lagopts.ind.X(2)), [N,K]);
        Y = reshape(x(Lagopts.ind.Y(1):Lagopts.ind.Y(2)), [K,Q]); 
        Z = reshape(x(Lagopts.ind.Z(1):Lagopts.ind.Z(2)), [N,Q]);
        W = reshape(x(Lagopts.ind.W(1):Lagopts.ind.W(2)), [N,Q]); 
        
        Lambdap = Lambda + (Z-X*Y)*rho;

        W_fro = norm(W, 'fro'); 
        Wf_iter = norm(W - X*Y, 'fro'); 
        OptTol = Wf_iter/W_fro; 
end
