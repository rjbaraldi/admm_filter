
function L = augLag(X, Y, Z, M, Lambda, rho)

L = .5*norm(Z - M, 'fro')^2 + trace(Lambda'*(Z - X*Y)) + rho/2*norm(Z - X*Y, 'fro')^2; 

end
