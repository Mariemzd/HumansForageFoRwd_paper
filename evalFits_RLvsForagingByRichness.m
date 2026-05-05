% Figure g-k 
clear; close all ; 

load('data/singleiti_202203011023_lightweight.mat') % 258 people on mTurk
load('data/fitRLtoMTurk_20rounds_220724_0322.mat')


%% first, a small correction, we want to save # choices for each row

nChoices = NaN(length(trials),1);
for k = 1:length(trials)
    % pull out the data
    tmp = trials(k).trials;
    selex = [trials(k).trials.practice]==0; % exclude practice trials
    nChoices(k) = sum(selex);
end

nSS = 254; % also just write down the number of subjects

% add richness 
richnessLL = [ 22439.687879281395, 23440.65905027464]; 
richnessAIC= [47419.37575856279, 48913.31810054928] ; 

%% Let's make some reference plots

blue = [67, 129, 193]./255;
% blue = [111, 156, 235]./255;
orange = [230, 142, 95]./255;

% we'll start just comparing vanilla RL vs vanilla Foraging

foragingIDs = find(strcmp({fits.modelName},'foraging'));
rlIDs = find(strcmp({fits.modelName},'standardRL'));

% just double check our likelihoods all line up b/c if some are missing...
tmp = arrayfun(@(k) fits(k).likelihood, [rlIDs,foragingIDs], 'UniformOutput', false);
selex = sum(isnan(horzcat(tmp{:})),2)==0;
wonk = [fits([rlIDs,foragingIDs]).likelihood];
logLik = nansum(wonk(selex,:)); % basically, we're only comparing datasets where all the models fit

sumNChoices = repmat(nansum(nChoices(selex)),size(logLik))
% logLik = [fits.lik]

figure(99); clf;
set(99,'Position',[476   446    1012    513]);
subplot(2,3,1);
plot(logLik,'.-k',...
    'MarkerSize',20)
set(gca,'FontSize',14,...           
    'XTick',1:length(logLik),...
    'XTickLabel',{fits([rlIDs,foragingIDs]).modelName})
xlim([0.5, length(logLik)+0.5])
tmp = ylim; ylim(tmp.*[0.99 1.01]);
ylabel('neg. log likelihood')
xlabel('model name')

-logLik
[aic] = aicbic(-logLik,[fits(([rlIDs,foragingIDs])).nParams].*nSS,sumNChoices)

subplot(2,3,4); hold all;
plot(aic,'.-',...
    'MarkerSize',20,'Color',[.3 .3 .3])
set(gca,'FontSize',14,...
    'XTick',1:length(logLik),...
    'XTickLabel',{fits([rlIDs,foragingIDs]).modelName})
xlim([0.5, length(logLik)+0.5])
tmp = ylim; ylim(tmp.*[0.99 1.01]);
ylabel('information criterion')
xlabel('model name')
legend('aic','bic','Location','NorthEast')

[~,bestAic] = min(aic)

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
aicW = exp(-(aic - min(aic))/2)


%% now, let's look at all the variations of RL + foraging

foragingIDs = find(cellfun(@isempty,strfind({fits.modelName},'RL')));
rlIDs = find(~cellfun(@isempty,strfind({fits.modelName},'RL')));

if length(foragingIDs) == length(rlIDs)
    xpos = 1:length(foragingIDs);
end

mStr = {'standard','asym','decay','choice+1','choice+2','richness'};

for plt = 1:2
    switch plt
        case 1
            IDs = rlIDs;
            cStr = blue;
            rLL = richnessLL(2) ; 
            rAIC = richnessAIC(2) ; 
        case 2
            IDs = foragingIDs;
            cStr = orange;
            rLL = richnessLL(1) ; 
            rAIC = richnessAIC(1) ;
    end

    % just double check our likelihoods all line up b/c if some are missing...
    tmp = arrayfun(@(k) fits(k).likelihood, [IDs], 'UniformOutput', false);
    selex = sum(isnan(horzcat(tmp{:})),2)==0;
    wonk = [fits([IDs]).likelihood];
    logLik = nansum(wonk(selex,:)); % basically, we're only comparing datasets where all the models fit
    
    sumNChoices = repmat(nansum(nChoices(selex)),size(logLik))
    % logLik = [fits.lik]
    zeb = [logLik,rLL] ; 
    subplot(1,3,2); hold on;
    plot(zeb,'.-','Color',cStr,...
        'MarkerSize',20)
    set(gca,'FontSize',20,...           
        'XTick',1:length(zeb),...
        'XTickLabel',mStr)
    xlim([0.5, length(zeb)+0.5])
    tmp = ylim; ylim(tmp.*[0.99 1.01]);

    ylabel('negative log likelihood')
    xlabel('model type')
    
    [aic] = aicbic(-logLik,[fits(([IDs])).nParams].*nSS,sumNChoices);
    subplot(1,3,3); hold all;
    zebi = [aic,rAIC] ; 
    plot(zebi,'.-',...
        'MarkerSize',20,'Color',[cStr.*0.9]);%./255)
    set(gca,'FontSize',20,...
        'XTick',1:length(zebi),...
        'XTickLabel',mStr, ...
        'YTick',[4.3:0.1:5.5]*10^4 )
    xlim([0.5, length(zebi)+0.5])
    ylim([4.7,5]*10^4)
    tmp = ylim; ylim([4.7, 5.05]*10^4 );
    ylabel('akaike information criterion (AIC)')
    xlabel('model type')
    legend('rl','f','Location','NorthEast')

end


subplot(1,3,2); legend('rl','foraging')

% finally, the ultimate comparison

% just double check our likelihoods all line up b/c if some are missing...
tmp = arrayfun(@(k) fits(k).likelihood, [foragingIDs,rlIDs], 'UniformOutput', false);
selex = sum(isnan(horzcat(tmp{:})),2)==0;
wonk = [fits([foragingIDs,rlIDs]).likelihood];
logLik = nansum(wonk(selex,:)); % basically, we're only comparing datasets where all the models fit

sumNChoices = repmat(nansum(nChoices(selex)),size(logLik));
% logLik = [fits.lik]

{fits([foragingIDs,rlIDs]).modelName}
logLik
[aic] = aicbic(-logLik,[fits(([foragingIDs,rlIDs])).nParams].*nSS,sumNChoices)

[~,bestAic] = min(aic)

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
aicW = exp(-(aic - min(aic))/2)

tmp = {fits(([foragingIDs,rlIDs])).modelName};
disp('best AIC')
tmp{bestAic}

% now pairwise comparison? is each version of foraging better than the
% corresponding RL model?
aicWpaired = exp(-(aic(length(aic)/2+1:end)-aic(1:length(aic)/2))/2)


%% now foraging vs the extensions of the RL model:

rforagingIDs = find(strcmp({fits.modelName},'foraging'));
rlIDs = find(and(~cellfun(@isempty,strfind({fits.modelName},'RL')),...
    cellfun(@isempty,strfind({fits.modelName},'standard'))));

IDs = [foragingIDs,rlIDs];

{fits(IDs).modelName}

% just double check our likelihoods all line up b/c if some are missing...
tmp = arrayfun(@(k) fits(k).likelihood, [IDs], 'UniformOutput', false);
selex = sum(isnan(horzcat(tmp{:})),2)==0;
wonk = [fits([IDs]).likelihood];
logLik = nansum(wonk(selex,:)) % basically, we're only comparing datasets where all the models fit

sumNChoices = repmat(nansum(nChoices(selex)),size(logLik));
% logLik = [fits.lik]

[aic] = aicbic(-logLik,[fits(([IDs])).nParams].*nSS,sumNChoices)

[~,bestAic] = min(aic)

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
aicW = exp(-(aic - min(aic))/2)

tmp = {fits(([foragingIDs,rlIDs])).modelName};
disp('best AIC')
tmp{bestAic}

% each compared only to foraging
aicWvF = exp(-(aic - aic(1))/2)

%% new foraging vs best RL


foragingIDs = find(~cellfun(@isempty,strfind({fits.modelName},'F')));
rlIDs = find(and(~cellfun(@isempty,strfind({fits.modelName},'RL')),...
    cellfun(@isempty,strfind({fits.modelName},'standard'))));
rlIDs = rlIDs(end);

IDs = [rlIDs,foragingIDs];

{fits(IDs).modelName}

% just double check our likelihoods all line up b/c if some are missing...
tmp = arrayfun(@(k) fits(k).likelihood, [IDs], 'UniformOutput', false);
selex = sum(isnan(horzcat(tmp{:})),2)==0;
wonk = [fits([IDs]).likelihood];
logLik = nansum(wonk(selex,:)) % basically, we're only comparing datasets where all the models fit

sumNChoices = repmat(nansum(nChoices(selex)),size(logLik));
% logLik = [fits.lik]

[aic] = aicbic(-logLik,[fits(([IDs])).nParams].*nSS,sumNChoices)

[~,bestAic] = min(aic)

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
aicW = exp(-(aic - min(aic))/2)

tmp = {fits(([foragingIDs,rlIDs])).modelName};
disp('best AIC')
tmp{bestAic}

% each compared only to foraging
aicWvF = exp(-(aic - aic(1))/2)

%% now we'll look on an individual basis, ask which fit better overall

foragingIDs = find(cellfun(@isempty,strfind({fits.modelName},'RL')));
rlIDs = find(~cellfun(@isempty,strfind({fits.modelName},'RL')));

IDs = [foragingIDs,rlIDs];

if length(foragingIDs) == length(rlIDs)
    xpos = 1:length(foragingIDs);
end

mStr = {'standard','asym','decay','choice+1','choice+2'};


% best fit via likelihood
fitMeasure = -[fits.likelihood]; % was negative log lik, now just log lik

% calculate the best model for each Ss
[~,bestFitIdx] = max(fitMeasure');

% select only good fits:
selex = sum(isnan([fits(IDs).agreement]),2)'==0; % meaning we didn't pre-select
% and the optimization terminated successfully
tmp = arrayfun(@(k) fits(k).params(:,end),IDs,'UniformOutput',false);
selex = and(selex,[sum(horzcat(tmp{:})==0,2)==0]');

% apply selection
fitMeasure = fitMeasure(selex,:);
bestFitIdx = bestFitIdx(selex);

disp('p(best fitting model):')
p_best = arrayfun(@(x) mean(bestFitIdx==x),[1:length(fits)])

 tmpMeasure = fitMeasure./300;

%%
for k = 1%,5]%,5]%length(foragingIDs)

    mStr{k}
    
    % median performance on this measure
    nanmedian([fitMeasure(:,foragingIDs(k)),fitMeasure(:,rlIDs(k))])
    quantile(fitMeasure(:,foragingIDs(k)),[0.025,0.975])
    quantile(fitMeasure(:,rlIDs(k)),[0.025,0.975])

    % fraction of subjects in which foraging does equal to or better:
    nanmean([fitMeasure(:,foragingIDs(k))-fitMeasure(:,rlIDs(k))]>=0)
    sum([fitMeasure(:,foragingIDs(k))-fitMeasure(:,rlIDs(k))]>=0)

    % average difference in our fit measure
    nanmean([fitMeasure(:,foragingIDs(k))-fitMeasure(:,rlIDs(k))])
    quantile([fitMeasure(:,foragingIDs(k))-fitMeasure(:,rlIDs(k))],[0.025,0.975])
        
    % now plots of the same
    figure(); clf;
    subplot(4,4,[5:7,9:11,12:15]); axis square; hold on;
    plot(aicbic(tmpMeasure(:,foragingIDs(k)),3),aicbic(tmpMeasure(:,rlIDs(k)),2),...
        '.','Color',[.3 .3 .3],'MarkerSize',10)
%     ylim([0.45 1])
    % tmp = [xlim,ylim]; xlim([min(tmp),max(tmp)]); ylim([min(tmp),max(tmp)]);
    % line([min(tmp) max(tmp)],[min(tmp) max(tmp)],'Color',[0 0 0])
    set(gca,'FontSize',14)
    ylabel({'average choice likelihood',strcat(mStr{k},' RL')})
    xlabel({'average choice likelihood',strcat(mStr{k},' foraging')})
    
    subplot(4,4,2:4);
    wonk = [tmpMeasure(:,foragingIDs(k))-tmpMeasure(:,rlIDs(k))];
    edges = -max(abs(wonk)):range(wonk)/10:max(abs(wonk));
    histogram(wonk,edges,'FaceColor',[.3 .3 .3],...
        'LineStyle','none')
    line([0 0],[ylim],'Color',[0 0 0])
    % xlim([-0.5 0.5]);
    % set(gca,'FontSize',10,'XTick',[-0.5,-0.25,0,0.25,0.5])

    % add reference line for -100/100 tick marks to make aligning easier
    subplot(4,4,[5:7,9:11,12:15]);
    % plot([min(tmp):0.1:max(tmp)],[min(tmp):0.1:max(tmp)]+0.25)
    % plot([min(tmp):0.1:max(tmp)],[min(tmp):0.1:max(tmp)]-0.25)

    % statistics on the difference:
    [h,p,ci,stat] = ttest(fitMeasure(:,foragingIDs(k)),fitMeasure(:,rlIDs(k))) % parametric
%     [p,h,ci] = signtest(fitMeasure(:,foragingIDs(k)),fitMeasure(:,rlIDs(k))) % non-parametric

    title(sprintf('p = %0.4f',p))

end

%% what if we compare the best foraging model with the best RL model within each Ss?

foragingIDs = find(cellfun(@isempty,strfind({fits.modelName},'RL')));
rlIDs = find(~cellfun(@isempty,strfind({fits.modelName},'RL')));

IDs = [foragingIDs,rlIDs];

if length(foragingIDs) == length(rlIDs)
    xpos = 1:length(foragingIDs);
end

mStr = {'standard','asym','decay','choice+1','choice+2'};

% best fit via likelihood
fitMeasure = -[fits.likelihood]; % maximize me

% calculate the best model for each Ss
[~,bestForIdx] = max(fitMeasure(:,foragingIDs)');
bestForIdx = foragingIDs(bestForIdx);
[~,bestRLIdx] = max(fitMeasure(:,rlIDs)');
bestRLIdx = foragingIDs(bestRLIdx);

% select only good fits:
selex = sum(isnan([fits(IDs).agreement]),2)'==0; % meaning we didn't pre-select
% and the optimization terminated successfully
tmp = arrayfun(@(k) fits(k).params(:,end),IDs,'UniformOutput',false);
selex = and(selex,[sum(horzcat(tmp{:})==0,2)==0]');

% apply selection
fitMeasure = fitMeasure(selex,:);
bestForIdx = bestForIdx(selex);
bestRLIdx = bestRLIdx(selex);

% convert our fit measure back to likelihood
tmpMeasure = exp(fitMeasure./300); % CAUTION!!!! MAKLE SURE I'M RIGHT
% tmpMeasure = fitMeasure;

% now evaluate, what's the quality of the best RL/F fit?
rlFits = tmpMeasure(sub2ind(size(tmpMeasure),[1:length(bestRLIdx)],bestRLIdx));
forFits = tmpMeasure(sub2ind(size(tmpMeasure),[1:length(bestForIdx)],bestForIdx));
nanmean(forFits>=rlFits)

[h,p,stat,ci] = ttest(forFits,rlFits)

