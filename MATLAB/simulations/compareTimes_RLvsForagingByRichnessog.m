%% how do we re-load the data now so we can work with it?

cd('/Users/becket/Documents/MATLAB/RLvsForagingByRichness/simulationOutputs')

files = dir;
fnames = {files(find(~cellfun(@isempty,strfind({files.name},'simResults')))).name};

for k = 1:length(fnames)
    load(fnames{k})

    if k == 1
        tmp = x;
    else
        tmp(:,k) = x;
    end
end

out = tmp;

%% let's estimate the distribution of our mixture model parameters
% using the 100ish samples that we took from each of the agents

nBoot = size(out,2);
nSs = 254; % how many people do we actually have
nParams = 3; % in mixture model

theta = NaN(nBoot,nParams+6,2); % preallocate

for model = 1:2
    switch model
        case 1
            selex = strcmp({out(:,1).type},'rl');
            tStr = 'rl'
        case 2
            selex = strcmp({out(:,1).type},'f');
            tStr = 'foraging'
    end

    goods = find(selex); % convert logical to scalar index
    
    for boot = 1:nBoot
        times = [out(goods,boot).times]-1;
        tmp = exp2mix(times);
       
%         switches = cellfun(@(x) nanmean(diff(x)~=0),{out(goods,boot).choices});
        % we'll pull all the switches together:
        switches = cellfun(@(x) (diff(x)~=0),{out(goods,boot).choices},'UniformOutput',false);
        switches = [switches{:}]; % later, we'll look at this on an individual level
        theta(boot,1:nParams,model) = [tmp(1:2),tmp(3)];
        theta(boot,nParams+1,model) = nanmean(times+1);
        theta(boot,nParams+2,model) = nanmean(switches);
        theta(boot,nParams+3,model) = nanmean(times+1>30); % 10% of trials
        theta(boot,nParams+4,model) = nanmean(times+1>60); % 20% of trials
        theta(boot,nParams+5,model) = nanmean([out(goods,boot).p_rwd]);
        theta(boot,nParams+6,model) = nanmean([out(goods,boot).p_best]);

    end

end

nanmean(theta,1)
nanstd(theta,1)


%% now pull in the human data

maxNComp = 2; % for the mixture models

if ~exist('trials')
    cd('/Users/becket/Documents/MATLAB/RLvsForagingByRichness')
    load('singleiti_202203011023_lightweight.mat')
end

% and do the same here
times = []; p_rwd = []; p_best = []; switches = [];

for k = 1:length(trials)
    % are we looking at non-practice trials and excluding the 4 people?
    selex = [trials(k).trials.practice]==0; % yep
    choices = [trials(k).trials(selex).choice]; % yep

    if length(unique(choices))>1
        times = [times, diff(find(diff(choices)~=0))-1];

        % for switches, we'll concatenate these all together and get the
        % mean p(switch) across all the participants choices
        switches = [switches, diff(choices)~=0];
        p_rwd = [p_rwd, nanmean([trials(k).trials(selex).reward])];

        % caculate p chose max
        V = [trials(k).header.reward_structure(:,selex)];
        p_best = [p_best,mean(V(sub2ind(size(V),choices+1,1:sum(selex)))==max(V))];
    end
end

figure('Position',[449  138  1061  472]);

subplot(1,6,1:3);
twoMixPlot(times,15)
title('human behavior')
ylim([0, 0.5]); xlim([0 15])
set(gca,'FontSize',14)

subplot(2,7,5:6);
twoMixLogPlot(times,100)

% now calculate the log likelihood and add that to the plot
loglik = NaN(1,maxNComp);
try
    for k = 1:maxNComp
        [~,tmp,loglik(k)] = discreteExpMix(times,k);
    end
end

subplot(2,7,[5:6]+7);
plot([1:maxNComp],loglik,'.-k','MarkerSize',20)
set(gca,'FontSize',16,'XTick',[1:maxNComp]);
xlim([0 maxNComp+1]); ylim([-3.5*10^4, -3*10^4])
ylabel('log likelihood')
xlabel('n components')

llr = -2*(loglik(1) - loglik(2));
df1 = 1;
df2 = 3;
df = df2 - df1;
pAdd2 = 1-chi2cdf(llr,df)

% subplot(1,6,5:6)
% xeval = [10,15,20,50];
% bar(arrayfun(@(x) nanmean(times>x),xeval));
% ylim([0 0.25]);
% set(gca,'FontSize',16,'XTick',[1:length(xeval)],...
%     'XTickLabel',num2str(xeval'));
% ylabel('p(run > length)')
% xlabel('length')

realTheta = exp2mix(times);
realTheta(1:2) = realTheta(1:2); % convert to switch p
realTheta(3) = realTheta(3); % convert to p(long)
realTheta(nParams+1) = nanmean(times+1); % run length
realTheta(nParams+2) = nanmean(switches); % p switch
realTheta(nParams+3) = nanmean(times+1 > 30); % 10%
realTheta(nParams+4) = nanmean(times+1 > 60); % 20%
realTheta(nParams+5) = nanmean(p_rwd);
realTheta(nParams+6) = nanmean(p_best);

%% now let's do some plots

pltStr = {'short t constant','long t constant','% short',...
    'mean run length','p(switch)','mean runs > 30','mean runs > 60',...
    'p(reward)','p(chose best)'};

figure('Position',[476   148   825   718]); hold on;
for k = 1:size(theta,2)
    subplot(3,3,k); hold on;
    violinPlot(squeeze(theta(:,k,:)))
    pH = plot([1,2],mean(squeeze(theta(:,k,:))),'.k','MarkerSize',30);
    % 70% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.15]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.85]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',3);
    % 95% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.025]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.975]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',0.5);
    line([0 3],[realTheta(k),realTheta(k)]);

    if k == 1
        ylim([1,1.35]);
    end

    set(gca,'FontSize',14,'XTick',[1,2],...
        'XTickLabel',{'RL','FOR'})
    ylabel(pltStr{k});
    xlabel('model');

    pltStr{k}
    mean(squeeze(theta(:,k,:)))
    quantile(squeeze(theta(:,k,:)),[0.025,0.975])

    realTheta(k)
    nanmean(realTheta(k)>squeeze(theta(:,k,:)))
end

%% does foraging do more of those 300 trial same choices?

for model = 1:2
    switch model
        case 1
            selex = find(strcmp({out.type},'rl'));
        case 2
            selex = find(strcmp({out.type},'f'));
    end

    tmp = arrayfun(@(k) length(unique(out(k).choices))==1,selex);

    length(tmp)
    sum(tmp)
    nanmean(tmp)

    % calculate the probability of at least the number we observed
    1-binocdf(4-1,258,nanmean(tmp))
end

% maybe we can make some kind of plot
nBoot = size(out,2);
result = NaN(nBoot,2);
for model = 1:2
    switch model
        case 1
            indx = find(strcmp({out(:,1).type},'rl'));
        case 2
            indx = find(strcmp({out(:,1).type},'f'));
    end

    for boot = 1:nBoot
        tmp = {out(indx,boot).choices};
        result(boot,model) = sum(arrayfun(@(k) length(unique(tmp{k}))==1,1:length(tmp)));
    end

end

figure('Position',[476   148   275   218]);
hold all; % try to keep this the same size as above

maxN = 10;
for k = 1:2
    xpos = sort([[0:maxN],[0:maxN]]);
    ypos = sort([cumsum(histc(result(:,k),[0:maxN]));[0;cumsum(histc(result(:,k),[0:maxN-1]))]]);
    plot(xpos,ypos)
end
xlim([-0.5 10.5]); ylim([0 105]);
plot([(4/258)*254],[88],'^k','MarkerFaceColor','k');
plot([(4/258)*254],[105],'vk','MarkerFaceColor','k');
% h = line([(4/258)*254,(4/258)*254],[0 100]);
% set(h,'Color','k')
set(gca,'FontSize',14)
ylabel('cumulative sum')
xlabel('# of whole-session runs')
legend('RL','foraging','observed','Location','SouthEast')

%% for the p(switch), mean run length, and mean # runs > 30, we can plot the distribution of people

nBoot = size(out,2);

% first, preallocate
[runLen,pSwitch,overThirty] = deal(NaN(size(out,1)./2,nBoot,2));
c = 1;

% now do the human participants
for k = 1:length(trials)
    % are we looking at non-practice trials and excluding the 4 people?
    selex = [trials(k).trials.practice]==0; % yep
    choices = [trials(k).trials(selex).choice]; % yep

    if length(unique(choices))>1

%         1./nanmean([diff(find(diff(choices)~=0))])
%         nanmean([diff(choices)~=0])

        pSwitch(c,:,1) = deal(nanmean([diff(choices)~=0]));
        runLen(c,:,1) = deal(nanmean([diff(find(diff(choices)~=0))]));
        overThirty(c,:,1) = deal(nanmean([diff(find(diff(choices)~=0))]>30));
        c = c+1;
    end
end

% now we will do the 2 models
for model = 1:2
    switch model
        case 1
            indx = find(strcmp({out(:,1).type},'rl'));
        case 2
            indx = find(strcmp({out(:,1).type},'f'));
    end

    for boot = 1:nBoot
        choices = {out(indx,boot).choices};

        pSwitch(:,boot,model+1) = arrayfun(@(k) nanmean(diff(choices{k})~=0),1:length(choices))';
        runLen(:,boot,model+1) = arrayfun(@(k) nanmean([diff(find(diff(choices{k})~=0))]),1:length(choices))';
        overThirty(:,boot,model+1) = arrayfun(@(k) nanmean([diff(find(diff(choices{k})~=0))]>30),1:length(choices))';
    end
end

% pSwitch = 1./runLen;

%%

behOI = pSwitch;

mdlStr = {'rl','foraging'};

figure();
for plt = 1:2
    subplot(1,2,plt); axis square; hold on;
    tmp = squeeze(nanmean(behOI,2));
    plot(tmp(:,plt+1),tmp(:,1),'.',...
        'MarkerSize',15,'Color',[.4 .4 .4])

    sse = nansum((behOI(:,:,1) - behOI(:,:,plt+1)).^2,1);

%     % cross entropy
%     crossH = -sum(behOI(:,:,1).*log(behOI(:,:,plt+1))+(1-behOI(:,:,1)).*log(1-behOI(:,:,plt+1)))

    set(gca,'FontSize',14)
    title(strcat(sprintf('sse = %2.4f +/- %2.4f, %2.4f',...
        nanmean(sse),quantile(sse,[0.025,0.975]))))
    
    tmp = [xlim,ylim];
    xlim([0 max(tmp)]); ylim([0 max(tmp)]);
    line([0 max(tmp)],[0 max(tmp)],'Color','k')
    xlabel(strcat(mdlStr{plt},' predicted p(switch)'))
    ylabel(strcat('observed p(switch)'))
end

% now let's do a quick stats test, is there sig. more error in the foraging
% than in the RL model when it comes to predicting p(switch)
tmp = squeeze(nanmean(behOI,2)); % pull the behavior back in
[h,p,ci,stat] = ttest((tmp(:,1)-tmp(:,2)).^2,(tmp(:,1)-tmp(:,3)).^2)

% now we want to ask if there's a systematic bias towards over/under
% estimation; first with RL:
nanmean((tmp(:,2)-tmp(:,1)))
nanstd((tmp(:,2)-tmp(:,1)))
[h,p,ci,stat] = ttest((tmp(:,2)-tmp(:,1)))

nanmean((tmp(:,3)-tmp(:,1)))
nanstd((tmp(:,3)-tmp(:,1)))
[h,p,ci,stat] = ttest((tmp(:,3)-tmp(:,1)))

%%

nParams = 4;
nBoot = size(out,2);
theta = NaN(nBoot,nParams,2); % preallocate

for model = 1:2
    switch model
        case 1
            indx = find(strcmp({out(:,1).type},'rl'));
        case 2
            indx = find(strcmp({out(:,1).type},'f'));
    end

    for boot = 1:nBoot
        % first, we have to reshape the data for our HMM code
        choices = {out(indx,boot).choices};
        rewards = {out(indx,boot).rewards}; % gotta reformat
        data = arrayfun(@(k) {choices{k};rewards{k}},1:length(choices),'UniformOutput',false);

       
        data = data(arrayfun(@(k) length(unique(data{k}{1,:}))>1,1:length(data)));

        % then run the HMM code and extract our parameters
        [ll, transmat, ~,~,emat] = fitOreOitHMM(data',false); % fit
        theta(boot,1,model) = transmat(1,1); % ore to ore
        theta(boot,2,model) = transmat(2,2); % oit to oit

        % now we can mess with the transition matrix
        tmp = stationaryDist(transmat);
        theta(boot,3,model) = tmp(1); % stability of ore
        theta(boot,4,model) = tmp(2); % stability of oit

    end
end

%% now same on the real peoples

realTheta = NaN(1,nParams);
data = cell(1,length(trials));
for k = 1:length(trials)
    selex = [trials(k).trials.practice]==0; % grab only practice
    choices = [trials(k).trials(selex).choice]+1; % pull out choices
    rewards = [trials(k).trials(selex).reward]; % pull out rewards;
    if length(unique(choices))>1
        data{k} = {choices;rewards};
    end
end

data = {data{arrayfun(@(k) ~isempty(data{k}),1:length(data))}};

% then run the HMM code and extract our parameters
[ll, transmat, ~,~,emat] = fitOreOitHMM(data',false); % fit
realTheta(1) = transmat(1,1); % ore to ore
realTheta(2) = transmat(2,2); % oit to oit

% now we can mess with the transition matrix
tmp = stationaryDist(transmat);
realTheta(3) = tmp(1); % stability of ore
realTheta(4) = tmp(2); % stability of oit

%% or we could just load the data b/c the above takes forever

load('HMMdynamics_compareTimes.mat')

%% multiply oit x 2 because there are 2 states

theta(:,4,:) = theta(:,4,:)*2
realTheta(4) = realTheta(4)*2

%% and now the plots

pltStr = {'ore2ore','oit2oit','stationary p(ore)','stationary p(oit)'};

figure('Position',[476   148   825   718]); hold on;
for k = 1:size(theta,2)
    subplot(3,3,k); hold on;
    violinPlot(squeeze(theta(:,k,:)))
    pH = plot([1,2],mean(squeeze(theta(:,k,:))),'.k','MarkerSize',30);
    % 70% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.15]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.85]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',3);
    % 95% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.025]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.975]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',0.5);
    line([0 3],[realTheta(k),realTheta(k)]);

    set(gca,'FontSize',14,'XTick',[1,2],...
        'XTickLabel',{'RL','FOR'})
    ylabel(pltStr{k});
    xlabel('model');

    pltStr{k}
    mean(squeeze(theta(:,k,:)))
    quantile(squeeze(theta(:,k,:)),[0.025,0.975])

    realTheta(k)
    nanmean(realTheta(k)>squeeze(theta(:,k,:)))
end

%% plot the potentials

figure(); hold all;
for plt = 1:3
    switch plt
        case 1
            tmp = realTheta;
        case 2
            tmp = geomean(theta(:,:,1)); % rl
        case 3
            tmp = geomean(theta(:,:,2)); % for
    end

    % we'll always illustrate it as oit, ore, so rectify inputs
    twoWellPotential_v2([1-tmp(3),tmp(3)],[1-tmp(2),1-tmp(1)])
    % set A to 0.18 in twoWellPotential_v2 for the nice illustration

%     threeWellPotential([1-tmp(3),tmp(3)],[1-tmp(2),1-tmp(1)])
end

set(gca,'FontSize',14,'XTick',[1,2],'XTickLabel',{'oit','ore'},...
    'YTick',[]);
ylabel('potential energy')
legend('participants','rl','foraging')
wonk = ylim;
ylim([wonk(1)-0.2, wonk(2)+0.3])


%% let's look at the times, maybe compare the stickiness of the real data against our mass of simulations

maxNComp = 2;

figure('Position',[476   330   604   536]);

for model = 1:2
    switch model
        case 1
            selex = strcmp({out.type},'rl');
            tStr = 'rl'
            ax1pos = [1:3];
            ax2pos = [5:6];
        case 2
            selex = strcmp({out.type},'f');
            tStr = 'foraging'
            ax1pos = [7:9];
            ax2pos = [11:12];
    end

    times = [[out(selex).times]-1]; % do the minus up front

    subplot(2,6,ax1pos);
    twoMixLogPlot(times)
    title(tStr)
    ylim([10.^-4, 10^0])

    % now calculate the log likelihood and add that to the plot
    loglik = NaN(1,maxNComp);
    try
        for k = 1:maxNComp
            [~,~,loglik(k)] = discreteExpMix(times,k);
        end
    end
    
    subplot(2,6,ax2pos);
    plot([1:maxNComp],loglik,'.-k','MarkerSize',20)
    set(gca,'FontSize',16,'XTick',[1:maxNComp]);
    xlim([0 maxNComp+1]); %ylim([-3.5*10^4, -3*10^4])
    ylabel('log likelihood')
    xlabel('n components')
    
    length(times)
    loglik
    llr = -2*(loglik(1) - loglik(2))
    df1 = 1;
    df2 = 3;
    df = df2 - df1;
    pAdd2 = 1-chi2cdf(llr,df)

end
