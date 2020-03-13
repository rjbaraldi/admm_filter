
function [L, L_g] = augLag(x, Lambda, Lagopts)
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

L = .5*norm(Z - W, 'fro')^2 + trace(Lambda'*(Z - X*Y)) + rho/2*norm(Z - X*Y, 'fro')^2;

nn = (Z-X*Y + Lambda/rho); 
gx = -rho*nn*Y'; 
gz = rho*nn + (Z - W); 
gy = -rho*X'*nn; 
gw = -(Z-W); 

L_g = [gx(:); gy(:); gz(:); gw(:)]; 


end
