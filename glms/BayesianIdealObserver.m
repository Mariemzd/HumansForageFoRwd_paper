function EV = BayesianIdealObserver(choice,reward) 
% take choice and reward and returns Expected value from a bayesian ideal
% observer
% MZ 2026

nTrials = length(choice);
nOptions = max(choice);


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

