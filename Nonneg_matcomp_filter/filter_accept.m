function [Filter, add_stat] = filter_accept(Filter,eta, omega, Lagopts)

%If there is an acceptable point, add it. Otherwise, return the same
%filter. 

beta = Lagopts.beta; 
gamma = Lagopts.gamma;


%grab all the points
eta_l = Filter(:,1); 
omega_l = Filter(:,2); 

%compute eta
% eta_stat = (eta-eta_l*beta<=0);
eta_stat = (eta>eta_l*beta); % should be all zeros if acceptable

%compate omega
% omega_stat = (omega<=omega_l - gamma*eta); 
omega_stat = (omega>omega_l - gamma*eta); %should be all zeros if acceptable 

if sum(eta_stat)==0 || sum(omega_stat)==0
    Filter = [Filter; [eta, omega]]; 
    add_stat = 1;
else
    %if not acceptable do nothing and return the same filter
    add_stat =0; 
    
end
    



end