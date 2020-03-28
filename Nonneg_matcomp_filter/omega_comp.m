function omega = omega_comp(x, Lambda, Lagopts)
%compute omega = ||min{x - l, max{x-u, \nabla_x Lp(x,y)}}||
    l = Lagopts.LB; 
    u = Lagopts.UB; 
    [~, L_grad] = augLag(x,Lambda, Lagopts);
    omega = min(x-l, max(x-u, L_grad)); 
    omega = norm(omega); 

end