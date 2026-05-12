%% explore vs exploit HMM fit + analyses 

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

        % unfortunately sometimes they don't explore, so we'll cut those
        % out - be good to come back and make this point later
        data = data(arrayfun(@(k) length(unique(data{k}{1,:}))>1,1:length(data)));

        % then run the HMM code and extract our parameters
        [ll, transmat, ~,~] = fitOreOitHMM(data',false); % fit
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
[ll, transmat, ~,~] = fitOreOitHMM(data',false); % fit
realTheta(1) = transmat(1,1); % ore to ore
realTheta(2) = transmat(2,2); % oit to oit

% now we can mess with the transition matrix
tmp = stationaryDist(transmat);
realTheta(3) = tmp(1); % stability of ore
realTheta(4) = tmp(2); % stability of oit
%% 
load('HMMdynamics_compareTimes.mat') % HMM fits can be found in figshare 

%% multiply oit x 2 because there are 2 states

theta(:,4,:) = theta(:,4,:)*2 ; 
realTheta(4) = realTheta(4)*2 ; 

%% and now the plots

pltStr = {'ore2ore','oit2oit','stationary p(ore)','stationary p(oit)'};

figure('Position',[476   148   825   718]); hold on;
for k = 1:size(theta,2)
    % keyboard
    subplot(3,3,k); hold on;
    violinPlot(squeeze(theta(:,k,:))) ; hold on ; 
    pH = plot([1,2],mean(squeeze(theta(:,k,:))),'.k','MarkerSize',30); hold on ; 
    % keyboard
    % 70% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.15]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.85]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',3); hold on;
    % 95% CIs
    eH = errorbar([1,2],mean(squeeze(theta(:,k,:))),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.025]),...
        mean(squeeze(theta(:,k,:)))-quantile(squeeze(theta(:,k,:)),[0.975]));
    set(eH,'CapSize',0,'LineStyle','none','Color','k',...
        'LineWidth',0.5); hold on;
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

    twoWellPotential([1-tmp(3),tmp(3)],[1-tmp(2),1-tmp(1)])

end

set(gca,'FontSize',14,'XTick',[1,2],'XTickLabel',{'oit','ore'},...
    'YTick',[]);
ylabel('potential energy')
legend('participants','rl','foraging')
wonk = ylim;
ylim([wonk(1)-0.2, wonk(2)+0.3])


