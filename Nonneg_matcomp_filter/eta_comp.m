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


