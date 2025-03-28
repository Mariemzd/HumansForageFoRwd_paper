function [ lik, agreement, v, p_exploit ] = model_dxForaging( data, params, maxBeta)
% fitRLforaging fits a foraging model that operates on the derivative
% rather than the estimated value - that is, this model calculates the
% change in the local rate of return to decide whether to explore or
% exploit

%% pull in all the model parameters
if isnan(params(4))
    [aW,aL] = deal(params(1));
else
    aW = params(1);
    aL = params(4);
end

if isnan(params(2))
    thresh = 0.5;
else
    thresh = params(2);
end

if nargin < 3
    maxBeta = 20;
end

b = params(3);
    % CAUTION: thresh is changed to be positive or negative
if (thresh >= -2 && thresh <= 2) && (aW >= 0 && aW <= 1) && (aL >= 0 && aL <= 1) && (b >= 0 && b < maxBeta)

    %% now pull in the data

    choice = data.choice;
    reward = data.reward;

    %% set up the testbed

    nArms = max(choice);
    nTrials = length(choice); % trials per session

    % preallocate
    [v,p_exploit,p_obs] = deal(NaN(1,nTrials));
    [p_choice] = deal(NaN(nArms,nTrials,2));

    % seed subjective values
    v(1) = deal(1); % seed at estimate of world rate  % CAUTION: changed to match RL, 7/29/2022, rbe; was deal(thresh)
    % seeding values at 0 makes them more exploratory
    if reward(1)==0; alpha = aL;
    else; alpha = aW; end
    v(2) = (v(1) + alpha.*(reward(1)-v(1)));

    % make some utilities
    % first, the value of exploitation
    mkChoice = @(v) 1 ./ (1 + exp(-(v-thresh).*b));

    % first observation, assuming explore is just 1/nArms
    p_exploit(1) = 1;
    p_choice(:,1,1) = 1/nArms; % p(choice | ore), t1
    p_choice(:,1,2) = 1/nArms; % p(choice | oit), t1
    p_obs(1) = 1/nArms; % b/c p_explore = 1

    % now run through the trials
    for t = 2:nTrials;

        % calculate the probability of explore/exploit
        p_exploit(t) = mkChoice(v(t)-v(t-1)); % based on derivative

        % p_choice under exploitation
        p_choice(:,t,2) = 0; % goes in the 2nd plate
        p_choice(choice(t-1),t,2) = 1;

        % p_choice under exploration
%         p_choice(:,t,1) = deal(1/nArms); % choose randomly
        % Alt. could choose randomly, but excluding the last option
        p_choice(:,t,1) = deal(1/(nArms-1)); % random btw non-matching
        p_choice(choice(t-1),t,1) = 0; % matching is set to 1

        % now we can calculate the lik of the observation
        p_obs(t) = ((1-p_exploit(t)).*p_choice(choice(t),t,1)) + ...
            (p_exploit(t).*p_choice(choice(t),t,2));

        if isnan(p_obs(t))
            keyboard()
        end

        if t < nTrials
            if reward(t)==0
                alpha = aL;
            else
                alpha = aW;
            end
                
            % now update the values, if they've explored, we re-seed
            if choice(t) ~= choice(t-1)
                v(t+1) = thresh + alpha.*(reward(t)-thresh);
            else % otherwise we update according to rpe
                v(t+1) = v(t) + alpha.*(reward(t)-v(t));
            end
        end
    end

    lik = sum(-log(p_obs));

    % thinking about calculating the model agreement
    choice_p = (1-p_exploit).*p_choice(:,:,1) + p_exploit.*p_choice(:,:,2);
    choice_p_max = max(choice_p);
    agreement = nanmean(choice_p(sub2ind(size(choice_p),choice,[1:length(choice)]))==choice_p_max);
    % this is the fraction of the time that the person chose the option
    % that the model thought was the most probable


else
    lik = Inf;
end
