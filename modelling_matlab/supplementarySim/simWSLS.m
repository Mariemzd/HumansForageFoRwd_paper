function [choice,rewarded,explore,v] = simWSLS( rl )
%simWSLS simulates observations from the RL model specified by the
%   input parameters rl
        
%% pull in all the model parameters

if (isfield(rl,'epsilon') && ~isempty(rl.epsilon))
    epsilon = rl.epsilon;
else
    epsilon = 0;
end

if (isfield(rl,'rewards') && ~isempty(rl.rewards)) % rewards
    V = rl.rewards;
    mkRwds = false;
else
    mkRwds = true;
    disp('no rewards given, generating rewards:')
    
    % we'll need some params to make a reward vector
    hazard = 0.4; % p of step
    stepSize = 0.4; % size of step
    rwdBounds = [0.1,0.9];
    rwdOpts = [rwdBounds(1):stepSize:rwdBounds(end)];
end
    
%% set up the testbed

nArms = rl.nArms;
nTrials = rl.nTrials; % trials per session

% preallocate
[v,choice,explore,rewarded] = deal(NaN(1,nTrials));


% seed objective values
if mkRwds
    [V] = deal(NaN(nArms,nTrials));
    V(:,1) = rwdOpts(randi(length(rwdOpts),1,nArms));
end

% now run the simulation
for t = 1:length(V)%nTrials; %MZ 2025, because a bug makes some peoples have less trials
    
    if t == 1
        choice(t) = randi(nArms); % choose randomly
    else % figure out if we switch or stay
        pStay = abs(rewarded(t-1)-epsilon);

        stay = rand < pStay; % decide if we stay

        if ~stay
            opts = pop([1:nArms],choice(t-1));
            choice(t) = opts(randi(nArms-1)); % so we will also
        else
            choice(t) = choice(t-1);
        end
    end
    
    % now observe the outcome of this choice
    rewarded(t) = rand < V(choice(t),t); % was called rwds
    
end

end
