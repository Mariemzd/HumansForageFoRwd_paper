%This code generates all stat panels for the first row of Figure 5
%(i.e. Fig 5 C,D,E) (AIC, switch probabilities as a function of predicted 
% probability of switch from simulations of the traditional RL model and 
% the foraging-RL model, and distribution of the difference in predicted 
% versus observed switch probability for both models).

% data, fits and simulations are on the figshare https://doi.org/10.6084/m9.figshare.32193990

%% get AIC for the Experiment 2 (fig 5C)
clear;
% cd('...\Experiment2_fits');
load('fitRLto_machingLawData_202205051012_20rounds_251111.mat');

num_models = length(fits);
aic_values = NaN(num_models, 1);

for i = 1:2 %only need Foraging and RL
    k = fits(i).nParams;
    nll = fits(i).lik;
    aic = 2*k*94 + 2*nll;
    aic_values(i) = aic;
    fits(i).AIC = aic;
    modelname{i} = fits(i).modelName
end
figure;
set(gcf, 'Position', [100, 100, 200, 350]);
plot([2,1],aic_values(1:2), '-o', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0, 0, 0],  'MarkerFaceColor', [0, 0, 0],...
    'Color', [0, 0, 0]);
    xlim([0.75, 2 + 0.25]);
    ylim([1.38*10^4 1.50*10^4])
    xticks(1:aic_values(1:2));
    set(gca, 'XTickLabel', flip(modelname), 'FontSize', 11)

    
%% aic weights
min_aic = min(aic_values(1:2));
delta_aic = aic_values(1:2) - min_aic;
relative_likelihood = exp(-0.5 * delta_aic);
sum_relative_likelihood = sum(relative_likelihood);
aic_weights = relative_likelihood / sum_relative_likelihood;

for i = 1:2
    fits(i).AIC_weight = aic_weights(i);
    fprintf('Model: %s, AIC Weight: %.4f\n', fits(i).modelName, fits(i).AIC_weight);
end

%% stats
aicF= 2*3 + 2*fits(1).likelihood ;
aicR= 2*2 + 2*fits(2).likelihood ;


% p = signrank(aicF,aicR);
[p, h, stats] = signrank(aicF, aicR, 'method', 'approximate');
[p, h, stats] = signrank(aicF, aicR);

z_score = stats.zval
W = stats.signedrank
p

median_aicF = nanmedian(aicF);
median_aicR = nanmedian(aicR);

median_nllF = nanmedian(fits(1).likelihood);
median_nllR = nanmedian(fits(2).likelihood);

differences = aicF - aicR;
n = sum(~isnan(differences) & differences ~= 0);
df_equivalent = n - 1;

mean_diff = nanmean(differences)
diff_range = [min(differences), max(differences)]

%% plot predicted p(switch) (Fig 5D)

% cd('...\simulationOutputs_Experiment2');

files = dir;
fnames = {files(find(~cellfun(@isempty,strfind({files.name},'simResults')))).name};
nmod = 2; %only rl and foraging 
name = 'MatchingLaw'
for k = 1:length(fnames)
    load(fnames{k})
    
    if k == 1
        tmp = out;
    else
        tmp(:,k) = out;
    end
end

out = tmp;

%% load data 
% cd('...\Experiment2_data');
load('machingLawData_202205051012.mat')
modCol = {[0.6, 0.5, 0.1],[0.0, 0.6, 0.2],[0, 0, 0]};

%% for the p(switch), mean run length, and mean # runs > 30, we can plot the distribution of people
nSs = 94; 
nBoot = size(out,2);

% first, preallocate
[runLen,pSwitch,overThirty] = deal(NaN(nSs,nBoot,2));
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

% behOI = runLen;
% behOI = overThirty;
behOI = pSwitch;

mdlStr = {'rl','foraging'};

N_subjects_RMSE = size(behOI, 1); 

figure();
for plt = 1:2
    subplot(1,2,plt); axis square; hold on;
    tmp = squeeze(nanmean(behOI,2));
    plot(tmp(:,plt+1),tmp(:,1),'.',...
        'MarkerSize',15,'Color',modCol{plt})

 %TOTAL SSE distribution (1 x 100 vector)
    sse_dist = nansum((behOI(:,:,1) - behOI(:,:,plt+1)).^2, 1);
    
    %MEAN SQUARED ERROR (MSE) distribution
    mse_dist = sse_dist / N_subjects_RMSE;
    
    %ROOT MEAN SQUARED ERROR (RMSE) distribution
    rmse_dist = sqrt(mse_dist);
    
    set(gca,'FontSize',14)
    
    title(sprintf('Model: %s\nMean RMSE = %2.4f (95%% CI: [%2.4f, %2.4f])',...
        mdlStr{plt}, nanmean(rmse_dist), quantile(rmse_dist, 0.025), quantile(rmse_dist, 0.975)))
    
    % set(gca,'FontSize',14)
    % title(strcat(sprintf('sse = %2.4f +/- %2.4f, %2.4f',...
    %     nanmean(sse),quantile(sse,[0.025,0.975]))))
    
    tmp = [xlim,ylim];
    xlim([0 max(tmp)]); ylim([0 max(tmp)]);
    line([0 max(tmp)],[0 max(tmp)],'Color','k')
    xlabel(strcat(mdlStr{plt},' predicted p(switch)'))
    ylabel(strcat('observed p(switch)'))
        xlim([0 1]);
        ylim([0 1]);
end

%% Non-Parametric Statistical Tests (Wilcoxon Signed-Rank Test)

tmp = squeeze(nanmean(behOI,2)); % pull the behavior back in
sqErr_RL = (tmp(:,1) - tmp(:,2)).^2;
sqErr_F = (tmp(:,1) - tmp(:,3)).^2;
err_RL = (tmp(:,2) - tmp(:,1));
err_F = (tmp(:,3) - tmp(:,1));

[p_sqErr, h_sqErr, stats_sqErr] = signrank(sqErr_RL, sqErr_F);

z_score = stats_sqErr.zval
W = stats_sqErr.signedrank
p_sqErr

differences = sqErr_RL - sqErr_F;
n = sum(~isnan(differences) & differences ~= 0);
df_equivalent = n - 1

mean_diff = nanmean(differences)
diff_range = [min(differences), max(differences)]

%% now to plot the distribution (figure 5E)
error_rl = tmp(:,2) - tmp(:,1);
error_foraging = tmp(:,3) - tmp(:,1);


figure;
set(gcf, 'Position', [100, 100, 600, 450]);
hold on;


bin_edges = min([error_rl; error_foraging]):0.02:max([error_rl; error_foraging]);
h1 = histogram(error_rl, bin_edges, 'FaceColor', modCol{1}, 'EdgeColor', 'k');
h1.FaceAlpha = 0.7;
h2 = histogram(error_foraging, bin_edges, 'FaceColor', modCol{2}, 'EdgeColor', 'k');
h2.FaceAlpha = 0.7;


yLims = ylim;


mean_err_rl = nanmean(error_rl);
mean_err_foraging = nanmean(error_foraging);
line([mean_err_rl mean_err_rl], yLims, 'Color', modCol{1}, 'LineStyle', '-', 'LineWidth', 3);
line([mean_err_foraging mean_err_foraging], yLims, 'Color', modCol{2}, 'LineStyle', '-', 'LineWidth', 3);

hold off;

set(gca, 'YAxisLocation', 'origin');

title('Overlapping Distribution of Prediction Errors', 'FontSize', 16);
xlabel('Prediction Error (Predicted - Observed)', 'FontSize', 12);
ylabel('Frequency (Subjects)', 'FontSize', 12);


legend(sprintf('RL Model (mean=%.3f)', mean_err_rl), ...
       sprintf('Foraging Model (mean=%.3f)', mean_err_foraging), ...
       'Zero Error', ...
       'Location', 'best');
       
set(gca, 'FontSize', 11);
xlim([-0.35 0.35])
box off;