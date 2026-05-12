function [ lik, agreement, data ] = model_WSLS( data, params, maxBeta)
%fitRLforaging fits a foraging-RL hybrid model

%% pull in all the model parameters
if isnan(params(1))
    epsilon = 0.1;
else
    epsilon = params(1);
end

if (epsilon >= 0 && epsilon <= 1)

    %% now pull in the data
    [lik,agreement] = deal(NaN(1,length(data)));
    
    % iterate
    for k = 1:length(data)
        choice = data(k).choice(:)';
        reward = data(k).reward(:)';

        %% set up the testbed
    
        nArms = max(choice);
        nTrials = length(choice); % trials per session
            
        % preallocate
        [p_obs] = deal(NaN(1,nTrials));
        [p_choice] = deal(NaN(nArms,nTrials));
    
        % first observation, we know nothing, so it's 1/nArms
        p_choice(:,1) = deal(1/nArms); % p(choice), t1
        p_obs(1) = 1/nArms; % b/c
         
        % now run through the trials
        for t = 2:nTrials;

            pStay = reward(t-1) * (1-epsilon) + epsilon/nArms ; %
        
            % calculate the probability of the choice
            p_choice(:,t) = deal((1-pStay)/(nArms-1)); % random btw non-matching
            p_choice(choice(t-1),t) = pStay; % matching is set to 1
            % keyboard 
            % now we can calculate the lik of the observation
            p_obs(t) = p_choice(choice(t),t);
    
            if isnan(p_obs(t))
                keyboard()
            end

        end

        lik(k) = sum(-log(p_obs));
    
        % thinking about calculating the model agreement
        choice_p_max = max(p_choice);
        agreement(k) = mean(p_choice(sub2ind(size(p_choice),choice,[1:length(choice)]))==choice_p_max,'omitnan');
        % this is the fraction of the time that the person chose the option
        % that the model thought was the most probable

    end

    lik = sum(lik);
    agreement = mean(agreement,'omitnan');

else
    lik = Inf;
end
