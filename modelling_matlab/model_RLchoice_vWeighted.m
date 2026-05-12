function [ lik, agreement, v, p_explore ] = model_RLchoice_vWeighted( data, params, maxBeta)
%model_RLchoice fits an RL model with choice history dependence

%% pull in all the model parameters
% params = alpha, beta, alphaCK, betaCK

a = deal(params(1)); % reward history learning
b = params(2); % reward history noise
ac = params(3); % choice history learning

if ~isnan(params(4))
    bc = params(4);
elseif ac == 0
    vW = 1; % if we've got no choice kernel, all weight to value
elseif ac ~= 0
    vW = 0.5; % otherwise, we take a weighted avg of choice + value
end

if nargin < 3
    maxBeta = 20;
end

% check the bounds on the paramters before we continue
if (a >= 0 && a <= 1) && (ac >= 0 && ac <= 1) && (b >= 0 && b < maxBeta) && (bc >= 0 && bc < maxBeta)

    %% now pull in the data

    choice = data.choice;
    reward = data.reward;
    
    %% set up the testbed
    nArms = max(choice);
    nTrials = length(choice); % trials per session

    % preallocate
    [p_obs] = deal(NaN(1,nTrials));
    [v,ck,p_choice] = deal(NaN(nArms,nTrials));

    % seed subjective values + choice histories
    v(:,1) = 0; % seed at estimate of world rate
    ck(:,1) = 0;

    % make some utilities
    mkChoice = @(v,ck) exp(b.*(vW.*v + (1-vW).*ck)) ./ sum(exp(b.*(vW.*v + (1-vW).*ck)));

    % now run through the trials
    for t = 1:nTrials

        % calculate the p_choice from the values
        p_choice(:,t) = mkChoice(v(:,t),ck(:,t));

        % now we can calculate the lik of the actual choice
        p_obs(t) = p_choice(choice(t),t);

        if isnan(p_obs(t))
            disp('p_obs is undefined')
            keyboard()
        end

        if t < nTrials
                
            % now update the values
            v(:,t+1) = v(:,t); % carry forward unchosen
            v(choice(t),t+1) = v(choice(t),t) + a.*(reward(t)-v(choice(t),t)); % update chosen

            % and update the choices
            ck(:,t+1) = ck(:,t) + ac.*(0-ck(:,t)); % update all like unchosen
            ck(choice(t),t+1) = ck(choice(t),t) + ac.*(1-ck(choice(t),t)); % then replace the chosen's update

        end
    end
    
    lik = sum(-log(p_obs));

    
    choice_p_max = max(p_choice);
    agreement = nanmean(p_choice(sub2ind(size(p_choice),choice,1:length(choice)))==choice_p_max);
    % this is the fraction of the time that the person chose the option
    % that the model thought was the most probable

else
    lik = Inf;
end
