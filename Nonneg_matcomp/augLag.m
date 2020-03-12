
function L = augLag(X, Y, Z,W, Lambda, rho)

L = .5*norm(Z - W, 'fro')^2 + trace(Lambda'*(Z - X*Y)) + rho/2*norm(Z - X*Y, 'fro')^2;

end
