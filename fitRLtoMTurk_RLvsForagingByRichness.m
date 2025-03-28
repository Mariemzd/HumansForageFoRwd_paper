clear;
%cd('/Users/mac/Desktop/Becket_models/')
load('/Users/mac/Desktop/GIT_RL_FORAGING_project/Mariem_Foraging_Q/matlab_data_2AB/humans_2armed_Ebitz_mTurk_State3_RwdProb.mat') % 258 people on mTurk

%%

nIter = 20; % number of iterations to fit for
maxBeta = 100; % highest value that beta can take on
rng(1); % seed random

% "preallocate"
fits = [];

for model = 12  %1:10
    switch model 
        case 1 % foraging
            nParams = 4; % how many parameters is the function expecting?
            nanParams = [0,0,0,1]; % logical, which to avoid fitting
            fits(model).modelName = 'foraging';
            betaSlot = 3;
        case 2 % vanilla RL
            nParams = 4;
            nanParams = [0,0,1,1];
            fits(model).modelName = 'standardRL';
            betaSlot = 2;
        case 3 % RL w/ asymetrical learning from wins and losses
            nParams = 4;
            nanParams = [0,0,0,1];
            fits(model).modelName = 'asymRL';
            betaSlot = 2;
        case 4 % RL w/ decaying value of unchosen option
            nParams = 4;
            nanParams = [0,0,1,0];
            fits(model).modelName = 'decayRL';
            betaSlot = 2;
        case 5 % RL w/ choice history kernel, 1 param = rate
            nParams = 4;
            nanParams = [0,0,0,1];
            fits(model).modelName = 'choiceRL+1';
            betaSlot = 2;
        case 6 % RL w/ choice history kernel, 2 param = beta + rate
            nParams = 4;
            nanParams = [0,0,0,0];
            fits(model).modelName = 'choiceRL+2';
            betaSlot = [2,4];
        case 7 % Foraging w/ asym rewards
            nParams = 7;
            nanParams = [0,0,0,0,1,1,1];
            fits(model).modelName = 'asymF';
            betaSlot = 3;
        case 8 % Foraging w/ decaying threshold
            nParams = 7;
            nanParams = [0,0,0,1,1,1,0];
            fits(model).modelName = 'decayF';
            betaSlot = 3;
        case 9 % Foraging w/ choice history, 1 param = rate
            nParams = 7;
            nanParams = [0,0,0,1,0,1,1];
            fits(model).modelName = 'choiceF+1';
            betaSlot = 3;
        case 10 % Foraging w/ choice history, 2 param = beta + rate
            nParams = 7; % slots
            nanParams = [0,0,0,1,0,0,1];
            fits(model).modelName = 'choiceF+2';
            betaSlot = [3,6];
        case 11 % Foraging w/ a bias term
            nParams = 4; % slots
            nanParams = [0,0,0,0]; % which to fit
            fits(model).modelName = 'biasedF';
            betaSlot = 3;
        case 12 % Foraging by the derivative
            nParams = 4; % slots
            nanParams = [0,0,0,1]; % which to fit
            fits(model).modelName = 'dxF';
            betaSlot = 3;
    end

    % spit the model name to the command line
    fits(model).modelName

    % initialize and pre-allocate
    fits(model).params = NaN(length(trials),nParams+5);
    fits(model).lik = 0;
    fits(model).nChoices = 0;
    fits(model).nParams = sum(nanParams==0);

    for k = 1:length(trials)

        % pull out the data
        tmp = trials(k).trials;
        selex = [trials(k).trials.practice]==0; % exclude practice trials
        t.choice = [tmp(selex).choice]+1;
        t.reward = [tmp(selex).reward];
        t.explore = [tmp(selex).states]==1;
    
        % pre-allocate model's parameters to be fitted
        mlpar = NaN(1,nParams+5);

        if length(unique([t.choice])) == 2 % if we get some choices

            % figure out what model to use
            switch model
                case 1
                    f = @(params) model_Foraging(t,params,maxBeta);
                case {5,6}
                    f = @(params) model_RLchoice(t,params,maxBeta);
                case {7,8,9,10}
                    f = @(params) model_ForagingFlex(t,params,maxBeta);
                case 11
                    f = @(params) model_ForagingBias(t,params,maxBeta);
                case 12
                    f = @(params) model_dxForaging(t,params,maxBeta);
                otherwise
                    f = @(params) model_RLflex(t,params,maxBeta); % last arg is max Beta
            end

            if k == 1
                f
            end
        
            % fitting options
            options = optimset('MaxFunEvals',2000,'MaxIter',1000,'Display','none',...
                'TolX',0.0001,'TolFun',0.0001);
        
            % initialize minima
            fvalmin = Inf;
        
            for i = 1:nIter
                % learning  noise
                initP = rand(1,nParams); % make a guess for this iteration
                initP(nanParams==1) = NaN; % don't fit the things we don't want to fit
                % CHANGE ME TO BE AN EXPONENTIAL DRAW
                initP(betaSlot) = initP(betaSlot).*maxBeta; % get beta closer to the right scale
        
                [guess fval exitflag output] = fminsearch(f,initP,options);
        
                if fval <= fvalmin && ~isinf(fval) && exitflag == 1 % save output if we get improvement
                    fvalmin = fval; % replace previous minima
                    [lik,agreement] = f(guess); % re-run the model
                    mlpar(1:nParams) = guess(1:nParams);
                    mlpar(nParams+2) = lik; % likelihood
                    mlpar(nParams+3) = agreement; % model agreement
                    mlpar(nParams+4) = output.iterations;
                    mlpar(nParams+5) = exitflag;
                end
            end
        
            fits(model).params(k,:) = mlpar;
            fits(model).lik = fits(model).lik + mlpar(nParams+2);
            fits(model).nChoices = fits(model).nChoices + length(t.choice);
        else
            disp(sprintf('too few choices, skipping participant %2.0f',k))
            fits(model).params(k,:) = mlpar;
        end
    end

    fits(model).avgParams = nanmean(fits(model).params);
    fits(model).likelihood = fits(model).params(:,nParams+2);
    fits(model).agreement = fits(model).params(:,nParams+3);
end

fname = strcat('fitRLtoMTurk_',datestr(datetime,'yymmdd_hhMM'));
save(fname,'fits')

%%

% just double check our likelihoods all line up b/c if some are missing...
tmp = arrayfun(@(k) fits(k).likelihood, [1:length(fits)], 'UniformOutput', false);
selex = sum(isnan(horzcat(tmp{:})),2)==0;
wonk = [fits.likelihood];
logLik = nansum(wonk(selex,:)); % basically, we're only comparing datasets where all the models fit

% logLik = [fits.lik]

figure(99); clf;
set(99,'Position',[476   446    1012    383]);
subplot(1,3,1);
plot(logLik,'.-k',...
    'MarkerSize',20)
set(gca,'FontSize',14,...           
    'XTick',1:length(fits),...
    'XTickLabel',{fits.modelName})
xlim([0.5, length(fits)+0.5])
ylabel('negative log likelihood')
xlabel('model name')

[aic,bic] = aicbic(-logLik,[fits.nParams],[fits.nChoices])
subplot(1,3,2); hold all;
plot(aic,'.-',...
    'MarkerSize',20)
plot(bic,'.-',...
    'MarkerSize',20)
set(gca,'FontSize',14,...
    'XTick',1:length(fits),...
    'XTickLabel',{fits.modelName})
xlim([0.5, length(fits)+0.5])
ylabel('information criterion')
xlabel('model name')
legend('aic','bic')

[~,bestAic] = min(aic)
[~,bestBic] = min(bic)

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
aicW = exp(-(aic - min(aic))/2)
bicW = exp(-(bic - min(bic))/2)

subplot(1,3,3); hold all;
m = cellfun(@(x) nanmean(x), {fits.agreement});
e = cellfun(@(x) nanstd(x)./sqrt(sum(~isnan(x))), {fits.agreement});
h = errorbar([1:length(fits)],m,e,...
    'CapSize',0,'Color','k','Marker','.',...
    'MarkerSize',20);
set(gca,'FontSize',14,...           
    'XTick',[1:length(fits)],...
    'XTickLabel',{fits.modelName})
xlim([0.5, length(fits)+0.5])
ylabel('model agreement')
xlabel('model name')

% save the output
saveas(99,fname,'epsc')
fname = 'messing'; % ensure we can't overwrite accidentally

%% now, on the individual level, how many were fit best by which model?

% best fit via likelihood
% fitMeasure = -[fits.likelihood];

% alt, best fit by model agreement?
fitMeasure = [fits.agreement];

% calculate the best model for each Ss
[~,bestFitIdx] = max(fitMeasure');

% select only good fits:
selex = sum(isnan([fits.agreement]),2)'==0; % meaning we didn't pre-select
% and the optimization terminated successfully
tmp = arrayfun(@(k) fits(k).params(:,end),[1:length(fits)],'UniformOutput',false);
selex = and(selex,[sum(horzcat(tmp{:})==0,2)==0]');

disp('p(best fitting model):')
p_best = arrayfun(@(x) mean(bestFitIdx(selex)==x),[1:length(fits)])

% we can also do some specific comparisons, like:
% 1) how often does foraging do better than vanilla RL?
nanmean(fitMeasure(:,1) > fitMeasure(:,2))

% 2) how often does foraging do better than *any* RL (i.e. the best RL)
p_best(1)

%% below here is just some questions about performance of the different models,
%   are people who are more RL-like better at the task?

disp('p(chose best | best fitting model):')
p_chose_best = arrayfun(@(k) trials(k).header.p_best, [1:length(trials)])';
arrayfun(@(x) mean(p_chose_best(and(selex,bestFitIdx==x))),[1:length(fits)])

disp('performance | best fitting model:')
p_rwd = arrayfun(@(k) trials(k).header.p_reward, [1:length(trials)])';
p_rwd_chance = arrayfun(@(k) trials(k).header.p_r_chance, [1:length(trials)])';
performance = (p_rwd ./ p_rwd_chance)-1;
arrayfun(@(x) mean(performance(and(selex,bestFitIdx==x))),[1:length(fits)])
