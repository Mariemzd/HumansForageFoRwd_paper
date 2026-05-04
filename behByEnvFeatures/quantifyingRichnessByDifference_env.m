clear; close all ; 

addpath('/Users/mac/Documents/MATLAB/general/')
datafile =  '../data/singleiti_202203011023_lightweight.mat' ; 
load(datafile)

% analyses found in figure 2 and supplementary figure S1C-G

%% we need some general parameters for the whole code right now

nBins = 3; % number of segments to divide the data into, 3 is 100 trial steps

purple = [0.4940 0.1840 0.5560];

%% first, we want to plot the the relationship between our 2 variables

out = NaN(length(trials)*nBins,2);

for k = 1:length(trials)
    
    tmp = trials(k).header.reward_structure;
    
    % richness is just the sum of the values
    richness = sum(trials(k).header.reward_structure);

    % sparsity as the range x1-x2 / the sum
    %  there's a perfect, nonlinear correlation between the two
    sparsity = range(trials(k).header.reward_structure)./sum(trials(k).header.reward_structure);
    
    stp = floor(length(richness)./nBins);
    
    for bin = 1:nBins-1
        out(k.*nBins-nBins+bin,1) = nanmean(richness(stp*bin-stp+1:stp*bin+stp));
        out(k.*nBins-nBins+bin,2) = nanmean(sparsity(stp*bin-stp+1:stp*bin+stp));
    end
end
out = out(sum(isnan(out),2)==0,:);

figure(96); set(gcf,'Position',[476   247   363   619])
subplot(2,1,1);
plot(out(:,1),out(:,2),'.k'); hold on;
[phat,s] = polyfit(out(:,1),out(:,2),2);
xpos = [0.4:0.05:1.6];
plot(xpos,polyval(phat,xpos),'LineWidth',3,'Color',purple);

[rho] = corr(out,'type','Spearman');
[R] = corr(out,'type','Pearson');
title(sprintf('Pearson R = %2.2f, Spearman rho = %2.2f',rho(1,2),R(1,2)))

set(gca,'FontSize',16)
ylabel('sparsity')
xlabel('richness')

subplot(2,1,2);
plot(out(:,2),out(:,1),'.k'); hold on;
[phat,s] = polyfit(out(:,2),out(:,1),1);
xpos = [0.1:0.05:0.7];
plot(xpos,polyval(phat,xpos),'LineWidth',3,'Color',purple);

set(gca,'FontSize',16)
xlabel('sparsity')
ylabel('richness')


% save the figure
%saveas(gcf,'richnessBySparsity_richnessByDifference','epsc')


%% now we want to plot both agents (or something like them) alongside the

figure(99); clf;
set(gcf,'Position',[440   578   658   480]);

% plotting stuff
hBins = 10; % for histogram

cmap = gray;
clist = [cmap(round(size(cmap,1).*0.1),:); ...
    cmap(round(size(cmap,1).*0.25),:); ...
    cmap(round(size(cmap,1).*0.4),:); ...
    cmap(round(size(cmap,1).*0.6),:); ...
    cmap(round(size(cmap,1).*0.75),:)];

for basis = 1:2
    switch basis
        case 1
            analysis = 'sparsity';
        case 2
            analysis = 'richness';
    end

    subplot(2,2,basis*2-1); hold on;

    tOpts = [0.05, 0.1, 0.2, 0.4];
    c = 1;

    lh = NaN(size(tOpts));
    
    for thresh = tOpts
    
        out = NaN(length(trials)*nBins,2);
    
        for k = 1:length(trials)
    
            tmp = trials(k).header.reward_structure;
            tmp = tmp(:,[trials(k).trials.practice]==0);
    
            richness = sum(tmp);
            difference = range(tmp);
            sparsity = range(tmp)./sum(tmp);
    
            if strcmp(analysis,'richness')
                x = richness; % richness as the predictor
                y = difference < thresh; %
                xlimits = [0 2.6];
            else
                x = sparsity;
                y = difference < thresh;
                xlimits = [-0.1 1.2];
            end
    
            stp = floor(length(x)./nBins);
    
            for bin = 1:nBins-1
                out(k.*nBins-nBins+bin,1) = nanmean(x(stp*bin-stp+1:stp*bin+stp));
                out(k.*nBins-nBins+bin,2) = nanmean(y(stp*bin-stp+1:stp*bin+stp));
            end
        end
        out = out(sum(isnan(out),2)==0,:);
    
        % plot raw
        %     plot(out(:,1),out(:,2),'.k');
        
        % else, let's do a histogram
        edges = quantile(out(:,1),[0:1/hBins:1]);
        [~,binidx] = histc(out(:,1),edges);
    
        m = arrayfun(@(x) nanmean(out(binidx==x,2)),[1:hBins]);
        e = arrayfun(@(x) nanste(out(binidx==x,2)),[1:hBins]);
    
        x = edges(1:end-1)+diff(edges)./2;
        lh(c) = plot(x,m,'.-','Color',clist(c,:),'MarkerSize',20);
        h = errbar(x,m,e); set(h,'Color',clist(c,:));

%         phat = polyfit(out(:,1),out(:,2),1);
%         xpos = [0:0.1:2.5];
%         lh(c) = plot(xpos,polyval(phat,xpos))
%         set(lh(c),'Color',clist(c,:));
        
        c = c+1;
    
    end
    
    set(gca,'FontSize',16)
    ylabel('p(diff < thresh)')
    xlabel(analysis)
    xlim(xlimits);
    ylim([0 1]);
    
    if basis == 1
        title('value-comparison')
    end
    legend(lh,cellstr(num2str(tOpts')))

    % now p(focal option) < some thresh?
    % this is what a foraging agent would do

    subplot(2,2,basis*2); hold on;
    
    c = 1;
    tOpts = [0.9,0.7,0.5,0.2]; % rho, parameter options
    lh = NaN(size(tOpts));
    
    for thresh = tOpts
    
        out = NaN(length(trials)*nBins,2);
    
        for k = 1:length(trials);
    
            tmp = trials(k).header.reward_structure;
            tmp = tmp(:,[trials(k).trials.practice]==0);
    
            if strcmp(analysis,'richness')
                x = sum(tmp); % richness
            else
                x = range(tmp)./sum(tmp); % difference/sparsity
            end
    
            most = nanmean([trials(k).header.reward_structure]<thresh); % best option
    
            stp = floor(length(richness)./nBins);
    
            for bin = 1:nBins-1
                out(k.*nBins-nBins+bin,1) = nanmean(x(stp*bin-stp+1:stp*bin+stp));
                out(k.*nBins-nBins+bin,2) = nanmean(most(stp*bin-stp+1:stp*bin+stp));
            end
        end
        out = out(sum(isnan(out),2)==0,:);
    
        % plot raw
        %     plot(out(:,1),out(:,2),'.k');
        
        % else, let's do a histogram
        edges = quantile(out(:,1),[0:1/hBins:1]);
        [~,binidx] = histc(out(:,1),edges);
    
        m = arrayfun(@(x) nanmean(out(binidx==x,2)),[1:hBins]);
        e = arrayfun(@(x) nanste(out(binidx==x,2)),[1:hBins]);
    
        x = edges(1:end-1)+diff(edges)./2;
        lh(c) = plot(x,m,'.-','Color',clist(c,:),'MarkerSize',20);
        h = errbar(x,m,e); set(h,'Color',clist(c,:));
    
%         phat = polyfit(out(:,1),out(:,2),1);
%         xpos = [0:0.1:2.5];
%         lh(c) = plot(xpos,polyval(phat,xpos))
%         set(lh(c),'Color',clist(c,:));
        
        c = c+1;
    
    end
    
    set(gca,'FontSize',16)
    ylabel('p(random < thresh)')
    xlabel(analysis)
    xlim(xlimits);
    ylim([0 1]);

    if basis == 1
        title('compare-to-threshold')
    end
    legend(lh,cellstr(num2str(tOpts')))

end

% save the figure
% saveas(gcf,'strategiesByRichnessAndSparsity_richnessByDifference','epsc')

%% distributions - for each agent, we want to extract its fingerprint
%   as a function of both sparsity and richness

glmDist = 'binomial';
% glmDist = 'normal';

rng(1);
nSims = 50;
figure(12); clf;

% for all the histograms, subplot 1:6
nHistBins = 20;
limits = {[0.85, 1.05],[-3.8, 3],[-2, 1],[0.75, 1.05],[-0.5 1.8],[-0.8 0.1]};
edges = cellfun(@(x) x(1):range(x)/nHistBins:x(end),limits,'UniformOutput',false);

for agent = 1:3
    switch agent
        case 1
            % first, we will look at the "RL"-like agent, who explores when the
            % difference in reward is less than some threshold
            maxDiff = 0.8;% max where you're not at ceiling
            minDiff = 0.05;% min where you're not at floor
%             tOpts = (rand(1,nSims) .* (maxDiff - minDiff)) + minDiff;
            tOpts = minDiff:0.1:maxDiff;
        case 2
            % now we want to look at what an Foraging-like agent would do
            % this is just asking whether any random arm is < the threshold
            maxThresh = 0.9; % max not at ceiling
            minThresh = 0.2; % min not at floor
%             tOpts = (rand(1,nSims) .* (maxThresh - minThresh)) + minThresh;
            tOpts = minThresh:0.1:maxThresh;
        case 3
            tOpts = 1; % only have 1 agent here
    end

    if agent == 1
        featureMx = NaN(length(tOpts),7,3);
    end

    c = 1;
    
    lh = NaN(size(tOpts));
    
    % iterate through the agents
    for thresh = tOpts
    
        out = NaN(length(trials)*nBins,2);
        tmpY = NaN(length(trials),1);
    
        % and make 1 agent for each session
        for k = 1:length(trials)
    
            tmp = trials(k).header.reward_structure;
            tmp = tmp(:,[trials(k).trials.practice]==0); % skip practice
    
            richness = sum(tmp);
            difference = range(tmp);
            sparsity = range(tmp)./sum(tmp);

            if agent == 1 % do RL-like switching
                y = difference < thresh; 
            elseif agent == 2 % do foraging-like switching
                y = nanmean(tmp <thresh); 
            elseif agent == 3 % real behavior
                y = [0, diff([trials(k).trials.choice])~=0]; % switches
                y = y([trials(k).trials.practice]==0); % skip practice
            end

            stp = floor(length(x)./nBins);
    
            for bin = 1:nBins-1
                out(k.*nBins-nBins+bin,1) = nanmean(sparsity(stp*bin-stp+1:stp*bin+stp));
                out(k.*nBins-nBins+bin,2) = nanmean(richness(stp*bin-stp+1:stp*bin+stp));
                out(k.*nBins-nBins+bin,3) = nanmean(y(stp*bin-stp+1:stp*bin+stp));
            end
    
            tmpY(k) = nanmean(y); % also calculate the p(switch)
        end
    
        out = out(sum(isnan(out),2)==0,:); % eliminate missing rows
    
        % so now we want to do some kind of quantitative comparison
        featureMx(c,1,agent) = nanmean(tmpY); % first feature is jus p(switch)
    
        dev = []; % now, we will look at sparsity (out(:,1))
        [l_phat,dev(1)] = glmfit([out(:,1)],out(:,end),glmDist); % linear
        [q_phat,dev(2)] = glmfit([out(:,1),out(:,1).^2],out(:,end),glmDist); % quadratic
    
        featureMx(c,2,agent) = dev(1)./dev(2);
        featureMx(c,3,agent) = q_phat(end); % quadratic component
        featureMx(c,4,agent) = l_phat(end); % linear component

        dev = []; % now, we will look at richness (out(:,2))
        [l_phat,dev(1)] = glmfit([out(:,2)],out(:,end),glmDist); % linear
        [q_phat,dev(2)] = glmfit([out(:,2),out(:,2).^2],out(:,end),glmDist); % quadratic
    
        featureMx(c,5,agent) = dev(1)./dev(2);
        featureMx(c,6,agent) = q_phat(end); % quadratic component
        featureMx(c,7,agent) = l_phat(end); % linear component

        c = c+1;
    
    end

    nanmean(featureMx(:,:,agent))

    %% plot the distributions
    figure(12);

    for k = 2:size(featureMx,2)
        subplot(2,3,k-1); axis square; hold on;
        if agent < 3
            ksdensity(featureMx(:,k,agent))
        else
            tmp = ylim;
            plot(featureMx(:,k,agent),tmp(2).*0.1,'vk','MarkerFaceColor','k')
        end
    end

end

% clean up this figure
figure(12);
subplot(2,3,6); hold on;
legend('rl','for','humans','Location','NorthWest')

aStr = {'linear/quad deviance','quadratic term','linear term'}; aStr = [aStr,aStr]
for k = 1:6
    subplot(2,3,k);
    set(gca,'FontSize',12)
    if k < 4
        title('sparsity')
    else
        title('richness')
    end
    xlabel(aStr{k});
end

%% now we want to do a PCA so we can try for some high-d thing
% the idea here is to take the idea of a fingerprint seriously - look at
% all the features simultaneously and ask which one is closer to the data

tmp = [featureMx(:,:,1);featureMx(:,:,2)]; % concatenate all the features from the simulations
tmp = tmp(sum(isnan(tmp),2)==0,:);

figure(13); clf;
subplot(2,1,1); axis square; hold all;
[coef] = pca(tmp,'Centered',false,...
    'NumComponents',2);
rl = featureMx(:,:,1)*coef; rl = rl(sum(isnan(rl),2)==0,:);
f = featureMx(:,:,2)*coef; f = f(sum(isnan(f),2)==0,:);
dat = featureMx(1,:,3)*coef;

ks = boundary(rl(:,1), rl(:,2));
pgon = polyshape(rl(ks(1:end-1),1),rl(ks(1:end-1),2));
pH = plot(pgon);
plot(rl(:,1), rl(:,2),'x','MarkerEdgeColor',[.5 .5 .5])

ks = boundary(f(:,1), f(:,2));
pgon = polyshape(f(ks(1:end-1),1),f(ks(1:end-1),2));
pH(2) = plot(pgon);
plot(f(:,1), f(:,2),'x','MarkerEdgeColor',[.5 .5 .5])

pH(3) = plot(dat(1),dat(2),'vk','MarkerFaceColor','k');
legend(pH,'rl','for','data','Location','NorthWest')

set(gca,'FontSize',12);
xlabel('PC1'); ylabel('PC2');

% now quantify the distances in the original d space
subplot(2,1,2); axis square; hold on;
ds1 = pdist2(unique(featureMx(:,:,1),'rows'),featureMx(1,:,3)); % rl
bar(1,nanmean(ds1),'BarWidth',0.75,'FaceAlpha',0.5)
plot(ones(size(ds1)),ds1,'x','MarkerEdgeColor',[.5 .5 .5])
ds2 = pdist2(unique(featureMx(:,:,2),'rows'),featureMx(1,:,3)); % for
bar(2,nanmean(ds2),'BarWidth',0.75,'FaceAlpha',0.5)
plot(ones(size(ds2)).*2,ds2,'x','MarkerEdgeColor',[.5 .5 .5])


n1 = length(ds1) ; n2 = length(ds2) ; 

if n1 == n2 
   [h,p,ci,stat] = ttest2(ds1,ds2);

   if h
    sprintf('avg. distance to rl = %2.2f +/- %2.2f STD',nanmean(ds1),nanstd(ds1))
    sprintf('avg. distance to foraging = %2.2f +/- %2.2f STD',nanmean(ds2),nanstd(ds2))
    sprintf('sig. difference in distances, p = %0.4f, t(%2.0f) = %2.2f',p,stat.df,stat.tstat)
   end

    
else 
    [p,h,stats] = ranksum(ds1,ds2) ;

    % effect size from % https://pmc.ncbi.nlm.nih.gov/articles/PMC12701665/
    num = stats.ranksum - (n1 * (n1 + 1)) / 2;
    den = sqrt(n1*n2*(n1+n2+1/12)) ;
    z = num/den ;
    r = z/sqrt(n1+n2) ;
    sprintf('med. distance to rl = %2.2f',nanmedian(ds1))
    sprintf('med. distance to foraging = %2.2f',nanmedian(ds2))
    sprintf('sig. difference in distances, W(%.0f) = %.0f, p = %.4f, r = %.2f', ...
    (n1 + n2), stats.ranksum, p, r)

end 


set(gca,'FontSize',12);
xlim([0 3]);
ylabel('distance');
xlabel('relative to')
set(gca,'XTick',[1,2],...
    'XTickLabels',{'rl','for'})