function EV = BayesianIdealObserver(choice,reward,static) 
% take choice and reward and returns Expected value from a bayesian ideal
% observer
% MZ 2026


if nargin<3
    static = true ; 
end 

 %hi Mariem

nTrials = length(choice);
nOptions = max(choice);

if static 

succ = NaN(nOptions,nTrials,2) ;
succ(:,1,:) = 0 ; %initialize
%first extract success and failure for each choice
for k=1:nTrials-1 % succ and failure will be used for the next trial 
 
    succ(choice(k),k+1,1)= any(reward(k)) ; % success
    succ(choice(k),k+1,2)= ~any(reward(k)) ; % failure
end


alpha = cumsum(succ(:,:,1),2,'omitnan')+1; %alpha = #success + 1
beta = cumsum(succ(:,:,2),2,'omitnan')+1; %alpha = #failures + 1


%compute Expected Value
EV = alpha ./(alpha+beta) ;

else 

%1st initialize 
mu0 = 1/2 ; 
k0 = 2 ; 
a0 = mu0*k0 ; 
b0 = k0*(1-mu0) ; 
p = 0.1 ; %p_step
delta = 0.1 ; %step_size
c  = -p*delta^2/2*pi^2 ; 
fh = @(x) x .* (1-x) ; 


mu = NaN(nOptions,nTrials,1) ; % mean belief
kappa = NaN(nOptions,nTrials,1) ; % concentration 
alpha = NaN(nOptions,nTrials,1) ; 
beta = NaN(nOptions,nTrials,1) ; 


mu(:,1) = mu0 ; 
kappa(:,1) = k0 ; 
alpha(:,1) = a0 ; 
beta(:,1) = b0 ; 

for t=1:nTrials-1
    
    % calculate mu and kappa for unchosen option everywhere
    mu(:,t+1) = 1/2 + exp(c)*(mu(:,t)-1/2) ; 
    num = fh(mu(:,t+1)) ; 
    den = fh(mu(:,t)) .\ kappa(:,t) + p*delta^2 ;

    kappa(:,t+1) =  max(k0,num./den); 
    alpha(:,t+1) =  mu(:,t+1).*kappa(:,t+1) ; 
    beta(:,t+1) = kappa(:,t+1) .* (1-mu(:,t+1)) ; 
    
    %then change val for chosen option 
    id = choice(t+1) ; 
    
    alpha(id,t+1) = alpha(id,t) + reward(t) ; 
    beta(id,t+1) =  beta(id,t) + (1-reward(t)) ; 
    kappa(id,t+1) = ( alpha(id,t+1) + beta(id,t+1) ) ; 

    mu(id,t+1) =  alpha(id,t+1) / kappa(id,t+1) ; 
     
end 

EV = mu ; 

end 
