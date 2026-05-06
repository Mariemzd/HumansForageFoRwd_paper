function [choices,rewarded,explore,v] = simRLvanilla( rl )
%quickSimRL simulates observations from the RL model specified by the
%   input in a 1-d (matching law) testbed

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

seed = 0.5;% CAUTION - seed for value

%% set up the testbed

nArms = rl.nArms;
nTrials = rl.nTrials;

nSessions = 1;

% preallocate
[choices,rewarded,explore] = deal(NaN(1,nTrials));
[v] = deal(NaN(nArms,nTrials));

% seed values
v(:,1) = deal(seed); % seed at 0

% utilities
pChoice = @(x) exp(x.*b)./sum(exp(x.*b));
if mkRwds
    [V] = deal(NaN(nArms,nTrials));
    V(:,1) = rwdOpts(randi(length(rwdOpts),1,nArms));
    updateRwd = @(V) max(min(V + (rand(nArms,1)<hazard).*(2.*(rand(nArms,1)<0.5)-1).*stepSize,...
        repmat(rwdBounds(2),nArms,1)),repmat(rwdBounds(1),nArms,1));
end

% now run the simulation
for t = 1:length(V)%nTrials;
    % make a choice, based on vdiff
    pCh = pChoice(v(:,t));
    choices(:,t) = find(rand < cumsum(pCh),1,'first');
    
    % draw obj. rewards for this choice
    % try
    tmp = V(choices(:,t),t); % use the real values
    % catch
    %     keyboard
    % end
    %  this works b/c matching law testbed
    
    rewarded(:,t) = rand(size(tmp)) < tmp; % check to see if they got a reward
    
    % update values
    if rewarded(:,t)
        alpha = aW;
    else
        alpha = aL;
    end

    v(choices(:,t),t+1) = v(choices(:,t),t) + ...
        alpha.*(rewarded(:,t) - v(choices(:,t),t));

    nonChoices = pop([1:nArms],choices(:,t));
    v(nonChoices,t+1) = v(nonChoices,t) - ...
        d.*v(nonChoices,t);
    
    % update rwd schedule for the next time step
    if t < nTrials
        if mkRwds
            V(:,t+1) = updateRwd(V(:,t));
        end
    end

end

%
xpos = [-1:0.1:1];
figure(98);
plot(xpos,pChoice(xpos))