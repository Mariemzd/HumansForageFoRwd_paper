function [choice,rewarded,explore,v] = simRLforaging( rl )
%simRLforaging simulates observations from the RL model specified by the
%   input parameters rl
        
%% pull in all the model parameters
if (isfield(rl,'alpha') || ~isempty(rl.alpha)) && ... % learning rates
        (~isfield(rl,'alphaAsym') || isempty(rl.alphaAsym))
    [aW,aL] = deal(rl.alpha);
elseif (isfield(rl,'alphaAsym') || ~isempty(rl.alphaAsym)) % alpha asymmetry
    aW = rl.alpha + rl.alphaAsym;
    aL = rl.alpha - rl.alphaAsym;
else
    [aW,aL] = deal(0.2);
end

if (isfield(rl,'beta') && ~isempty(rl.beta)) % beta/softmax temp
    b = rl.beta;
else
    b = 5;
end

if (isfield(rl,'delta') && ~isempty(rl.delta)) % decay of unchosen
    d = rl.delta;
else
    d = 0;
end

if (isfield(rl,'thresh') && ~isempty(rl.thresh)) % decay of unchosen
    thresh = rl.thresh;
else
    thresh = 0; 
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

% seed subjective values
v(1) = deal(thresh); % seed at estimate of world rate

% seed objective values
if mkRwds
    [V] = deal(NaN(nArms,nTrials));
    V(:,1) = rwdOpts(randi(length(rwdOpts),1,nArms));
end

% make some utilities
mkChoice = @(v) 1 ./ (1 + exp((v-thresh).*b)); % softmax for choosing
if mkRwds
    updateRwd = @(V) max(min(V + (rand(nArms,1)<hazard).*(2.*(rand(nArms,1)<0.5)-1).*stepSize,...
        repmat(rwdBounds(2),nArms,1)),repmat(rwdBounds(1),nArms,1));
end

% now run the simulation
for t = 1:length(V)%nTrials;
    
    % make a choice to explore/exploit
    if t > 1
        explore(t) = rand < mkChoice(v(t));
    else
        explore(t) = 1;
    end
    
    if explore(t)
        if t == 1
            choice(t) = randi(nArms); % choose randomly
        else % we're doing exclusive choosing now!
            
            opts = pop([1:nArms],choice(t-1));
            choice(t) = opts(randi(nArms-1)); % so we will also
            
        end
        v(t) = thresh; % re-seed the value
    else
        choice(t) = choice(t-1); % repeat the last choice
    end
    
    % now observe the outcome of this choice

    rewarded(t) = rand < V(choice(t),t); % was called rwds

  
    
    if rewarded(t)==0
        alpha = aL;
    else
        alpha = aW;
    end
    
    if t < nTrials
        % update subjective value for the next trial
        v(t+1) = v(t) + alpha.*(rewarded(t)-v(t));

        % update objective values for the next trial if needed
        if mkRwds
            V(:,t+1) = updateRwd(V(:,t));
        end
    end
    
end

end
