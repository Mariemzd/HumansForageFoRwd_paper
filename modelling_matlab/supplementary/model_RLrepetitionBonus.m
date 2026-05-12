function [ lik, agreement, v, p_explore ] = model_RLrepetitionBonus( data, params, maxBeta)
%model_RLchoice fits an RL model with repetition bonus

%% pull in all the model parameters
% params = alpha, beta, alphaCK, betaCK

a = deal(params(1)); % reward history learning
rb = params(2); % repetition bonus
b = params(3); % reward history noise

if nargin < 3
    maxBeta = 100;
end

% check the bounds on the paramters before we continue
if (a >= 0 && a <= 1) && (rb >= 0 && rb <= 1) && (b >= 0 && b < maxBeta)

    %% now pull in the data
    [lik,agreement] = deal(NaN(1,length(data)));

    for k = 1:length(data)

        choice = data(k).choice(:)';
        reward = data(k).reward(:)';

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
        mkChoice = @(v,ck) exp( ((1-rb).*v+ ck.*rb).*b) ./ sum(exp( ((1-rb).*v+ ck.*rb).*b));
         
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
                ck(:,t+1) = 0; % make all unchosen == 0
                ck(choice(t),t+1) = 1; % then replace the chosen w/ 1
    
            end
        end
        
        lik(k) = sum(-log(p_obs));
    
        % thinking about calculating the model agreement
        choice_p_max = max(p_choice);
        agreement(k) = nanmean(p_choice(sub2ind(size(p_choice),choice,[1:length(choice)]))==choice_p_max);
        % this is the fraction of the time that the person chose the option
        % that the model thought was the most probable

        % save the values
        data(k).v = v;
    end

    lik = sum(lik);
    agreement = nanmean(agreement);
else
    lik = Inf;
end
