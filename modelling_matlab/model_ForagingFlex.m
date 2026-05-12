function [ lik, agreement, v, p_explore ] = model_Foraging( data, params, maxBeta)
%fitRLforaging fits a foraging-RL hybrid model

%% pull in all the model parameters
if isnan(params(4))
    [aW,aL] = deal(params(1));
else
    aW = params(1);
    aL = params(4);
end

if ~isnan(params(5))
    ac = params(5);
else
    ac = 0; % no choice history learning
end

b = params(3);

if ~isnan(params(6))
    bc = params(6);
else
    bc = b; % otherwise, we take a weighted avg of choice + value
end

if ~isnan(params(7))
    decay = params(7);
else
    decay = 1;
end

if ~isnan(params(2))
    thresh = params(2);
else
    thresh = 0.5;
end

if nargin < 3
    maxBeta = 20;
end

                    % CAUTION: max is at 2, not 1
if (thresh >= 0 && thresh <= 2) && (aW >= 0 && aW <= 1) && (aL >= 0 && aL <= 1) && (b >= 0 && b < maxBeta) && (ac >= 0 && ac <= 1) && (bc >= 0 && bc <= maxBeta) && (decay >= 0 && decay <= 1)

    %% now pull in the data

    choice = data.choice;
    reward = data.reward;

    %% set up the testbed

    nArms = max(choice);
    nTrials = length(choice); % trials per session

    % preallocate
    [v,ck,p_explore,p_obs] = deal(NaN(1,nTrials));
    [p_choice] = deal(NaN(nArms,nTrials,2));

    % seed subjective values
    v(1) = deal(1); % seed at estimate of world rate % CAUTION: changed to match RL, 7/29/2022, rbe; was deal(thresh)
    % seeding values at 0 makes them more exploratory
    if reward(1)==0; alpha = aL;
    else; alpha = aW; end
    v(2) = (v(1) + alpha.*(reward(1)-v(1)));

    % seed choice histories
    ck(1:2) = deal(0); % ck is initalized at 0 for the first 2 trials
    % because the first trial where we know if the last pair was switch or
    % stay is trial 3

    % initialize local threshold
    l_thresh = thresh; % CAUTION, I'M NOT SURE THIS IS RIGHT FOR TR 2

    % make some utilities
    mkChoice = @(v,ck,thresh) 1 ./ (1 + exp(b.*(v-thresh) + bc.*ck));

    % first observation, assuming explore is just 1/nArms
    p_explore(1) = 1;
    p_choice(:,1,1) = 1/nArms; % p(choice | ore), t1
    p_choice(:,1,2) = 1/nArms; % p(choice | oit), t1
    p_obs(1) = 1/nArms; % b/c p_explore = 1

    % now run through the trials
    for t = 2:nTrials

        % calculate the probability of explore/exploit
        p_explore(t) = mkChoice(v(t),ck(t),l_thresh);

        % p_choice under exploitation
        p_choice(:,t,2) = 0;
        p_choice(choice(t-1),t,2) = 1;

        % p_choice under exploration
%         p_choice(:,t,1) = deal(1/nArms); % choose randomly
        % Alt. could choose randomly, but excluding the last option
        p_choice(:,t,1) = deal(1/(nArms-1)); % random btw non-matching
        p_choice(choice(t-1),t,1) = 0; % matching is set to 1

        % now we can calculate the lik of the observation
        p_obs(t) = (p_explore(t).*p_choice(choice(t),t,1)) + ...
            ((1-p_explore(t)).*p_choice(choice(t),t,2));

        if isnan(p_obs(t))
            keyboard()
        end

        if t < nTrials
            if reward(t)==0
                alpha = aL;
            else
                alpha = aW;
            end
                
            % now update the values + choice histories + threshold
            if choice(t) ~= choice(t-1) % if they've explored, we re-seed
                l_thresh = thresh;
                v(t+1) = l_thresh + alpha.*(reward(t)-l_thresh);
                ck(t+1) = 0; % reset ck to initial value
            else % otherwise we update everything
                l_thresh = l_thresh.*decay; % decay threshold
                v(t+1) = v(t) + alpha.*(reward(t)-v(t)); % reward rpe
                ck(t+1) = ck(t) + ac.*(1-ck(t)); % and increment ck
            end
        end
    end

    lik = sum(-log(p_obs));

    % thinking about calculating the model agreement
    choice_p = p_explore.*p_choice(:,:,1) + (1-p_explore).*p_choice(:,:,2);
    choice_p_max = max(choice_p);
    agreement = nanmean(choice_p(sub2ind(size(choice_p),choice,[1:length(choice)]))==choice_p_max);
    % this is the fraction of the time that the person chose the option
    % that the model thought was the most probable

else
    lik = Inf;
end
