function [ lik, agreement, v, p_explore ] = model_RLflex( data, params, maxBeta)
%model_RLflex fits an RL model with some flexibility in terms of what it
%does, can fit a vanilla 2-param model, a model w/ asymetrical learning
% (param3), and/or decaying the value of unchosen options (param4)
%   leave the parameters as NaNs if you don't want to fit them

%% pull in all the model parameters
% params = alpha(W), beta, alphaL

if isnan(params(3))
    [aW,aL] = deal(params(1));
else
    aW = params(1); % CAUTION, I AM NOT IMPLEMENTED LIKE FORAGING
    aL = params(3);
end

if isnan(params(4))
    decay = 1;
else
    decay = params(4);
end

b = params(2);

if nargin < 3
    maxBeta = 20;
end

% check the bounds on the paramters before we continue
if (aW >= 0 && aW <= 1) && (aL >= 0 && aL <= 1) && (decay >= 0 && decay <= 1) && (b >= 0 && b < maxBeta)

    %% now pull in the data

    choice = data.choice;
    reward = data.reward;
    
    %% set up the testbed
    nArms = max(choice);
    nTrials = length(choice); % trials per session

    % preallocate
    [p_obs] = deal(NaN(1,nTrials));
    [v,p_choice] = deal(NaN(nArms,nTrials));

    % seed subjective values
    v(:,1) = 0; % seed at estimate of world rate

    % make some utilities
    mkChoice = @(v) exp(v.*b) ./ sum(exp(v.*b));

    % first observation, assuming explore is just 1/nArms
    p_obs(1) = 1/nArms; % b/c p_explore = 1

    % now run through the trials
    for t = 1:nTrials

        % calculate the p_choice from the values
        p_choice(:,t) = mkChoice(v(:,t));

        % now we can calculate the lik of the actual choice
        p_obs(t) = p_choice(choice(t),t);

        if isnan(p_obs(t))
            disp('p_obs is undefined')
            keyboard()
        end

        if t < nTrials
            if reward(t)==0
                alpha = aL;
            else
                alpha = aW;
            end
                
            % now update the values
            v(:,t+1) = decay.*v(:,t); % carry forward unchosen, but decay if that param exists
            v(choice(t),t+1) = v(choice(t),t) + alpha.*(reward(t)-v(choice(t),t)); % update chosen
        end
    end
    
    lik = sum(-log(p_obs));

    % thinking about calculating the model agreement
    choice_p_max = max(p_choice);
    agreement = nanmean(p_choice(sub2ind(size(p_choice),choice,[1:length(choice)]))==choice_p_max);
    % this is the fraction of the time that the person chose the option
    % that the model thought was the most probable

else
    lik = Inf;
end
