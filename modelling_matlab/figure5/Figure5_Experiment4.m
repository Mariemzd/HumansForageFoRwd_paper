%This code generates all stat panels for the first row of Figure 5
%(i.e. Fig 5 L,M,O, P) (AIC, switch probabilities as a function of predicted 
% probability of switch from simulations of the traditional RL model and 
% the foraging-RL model, and distribution of the difference in predicted 
% versus observed switch probability for both models).

% data, fits and simulations are on the figshare https://doi.org/10.6084/m9.figshare.32193990
%% Get the walk from the bandit task for each subgroup (2-arm, 3-arm, 4-arm).
clear ; close all ; 

% cd('...\Experiment4_data');

%load 1 of the 3 datasets to plot the walks 

% load('cardBandit2arm.mat')
% load('cardBandit3arm.mat')
load('cardBandit4arm.mat')


num_trials = 300;

    participant_1_trials = trials(4).trials; 
    num_trials = length(participant_1_trials);
    num_participants = length(trials);
    
    reward_seed_option1 = zeros(num_trials, 1);
    reward_seed_option2 = zeros(num_trials, 1);
    reward_seed_option3 = zeros(num_trials, 1);
    reward_seed_option4 = zeros(num_trials, 1);
    

    for N = 1:num_trials
        current_seeds = participant_1_trials(N).reward_seed;
        reward_seed_option1(N) = current_seeds(1);
        reward_seed_option2(N) = current_seeds(2);
        reward_seed_option3(N) = current_seeds(3);
        reward_seed_option4(N) = current_seeds(4);
    end
    
    figure('Name', 'Bandit Reward Seeds Over Trials (Participant 1)');
    plot(reward_seed_option1, 'LineWidth', 2, 'DisplayName', 'Option 1 Reward Seed', 'Color', [0.0 0.45 0.74]);
    hold on;
    plot(reward_seed_option2, 'LineWidth', 2, 'Color', [0.85 0.33 0.10], 'DisplayName', 'Option 2 Reward Seed');
    hold on;
    plot(reward_seed_option3, 'LineWidth', 2, 'Color', [0.25 0.23 0.10], 'DisplayName', 'Option 3 Reward Seed');
    hold on;
    plot(reward_seed_option4, 'LineWidth', 2, 'Color', [0.85 0.23 0.60], 'DisplayName', 'Option 3 Reward Seed');

    set(gca, 'FontSize', 14, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02]);
    xlim([0 50]);

    disp('Plot generated successfully.');


%% get AIC plot

clear ; close all ; 


% cd('...\Experiment4_fits')

%load fits
TWOarmed = load('fitRLto_cardBandit2arm.mat_20rounds_251112.mat');
THREEarmed = load('fitRLto_cardBandit3arm.mat_20rounds_251112.mat');
FOURarmed = load('fitRLto_cardBandit4arm.mat_20rounds_251112.mat');


D = {TWOarmed.fits, THREEarmed.fits, FOURarmed.fits};
F = [50; 47; 50];
AIC = NaN(2, 3);
for d = 1:3
    for m = 1:2
        k = D{d}(m).nParams;
        log_likelihood = D{d}(m).lik;
        AIC(m, d) = 2*F(d)*(k) + 2*log_likelihood;
        D{d}(m).AIC = AIC(m, d);
    end
end

modelnames = {D{1}(1).modelName, D{1}(2).modelName};
figure;
set(gcf, 'Position', [100, 100, 200, 350]);
hold on;
% Define x_coords in increasing order: [1] for RL, [2] for Foraging
x_coords = [1, 2];
% flipud to plot AIC_RL at x=1 and AIC_Foraging at x=2
plot(x_coords, flipud(AIC(:, 1)), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0, 0, 0], 'MarkerFaceColor', [0, 0, 0]);
plot(x_coords, flipud(AIC(:, 2)), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.7, 0, 0.7], 'MarkerFaceColor', [0.7, 0, 0.7]);
plot(x_coords, flipud(AIC(:, 3)), '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0, 0.5, 0], 'MarkerFaceColor', [0, 0.5, 0]);
xlim([0.75, 2.25]);

xticks(x_coords);

set(gca, 'XTickLabel', flip(modelnames), 'FontSize', 11);
ylabel('AIC Value');
title('Model AIC Comparison');
legend('2-Arm', '3-Arm', '4-Arm', 'Location', 'best');
box off;
hold off;
%% aic weight calculation
AIC_Weights_Summary = NaN(2, 3); 
condition_names = {'2-Arm Bandit', '3-Arm Bandit', '4-Arm Bandit'};

for d = 1:3 
    
    current_AIC = AIC(:, d); 
    
    % minimum AIC (best model)
    min_aic = min(current_AIC);
    
    % Delta AIC (difference from the minimum)
    delta_aic = current_AIC - min_aic;
    
    % Relative Likelihood (L_i)
    relative_likelihood = exp(-0.5 * delta_aic);
    
    % AIC Weights (w_i)
    sum_relative_likelihood = sum(relative_likelihood);
    aic_weights = relative_likelihood / sum_relative_likelihood;
    
    for m = 1:2
        D{d}(m).AIC_weight = aic_weights(m);
        AIC_Weights_Summary(m, d) = aic_weights(m);
    end
    
    % results
    fprintf('\\n--- Condition: %s ---\\n', condition_names{d});
    for m = 1:2
        fprintf('Model: %10s, AIC Weight (w_i): %.4f\\n', D{d}(m).modelName, D{d}(m).AIC_weight);
    end
end

%% wilcoxon test individual diff


nllF_all = [TWOarmed.fits(1).likelihood(:); ...
            THREEarmed.fits(1).likelihood(:); ...
            FOURarmed.fits(1).likelihood(:)];


nllR_all = [TWOarmed.fits(2).likelihood(:); ...
            THREEarmed.fits(2).likelihood(:); ...
            FOURarmed.fits(2).likelihood(:)];


aicF_all = 2*3 + 2*nllF_all;
aicR_all = 2*2 + 2*nllR_all;

[p, h, stats] = signrank(aicF_all, aicR_all);

%Z and W statistics
z_stat = stats.zval;       
w_stat = stats.signedrank 
p

differences = aicF_all - aicR_all;
n = sum(~isnan(differences) & differences ~= 0);
df_equivalent = n - 1

mean_diff = nanmean(differences)
diff_range = [min(differences), max(differences)]

median_aicF = nanmedian(aicF_all)
median_aicR = nanmedian(aicR_all)
median_nllF = nanmedian(nllF_all)
median_nllR = nanmedian(nllR_all)

%% Load the simulation data 
clear; clc;

%load 1 at the time for this piece of code

% cd('...simulationOutput_Experiment4\2arm');
% cd('...simulationOutput_Experiment4\3arm');
% cd('...simulationOutput_Experiment4\4arm');



files = dir;
fnames = {files(find(~cellfun(@isempty,strfind({files.name},'simResults')))).name};
nmod = 2; %only rl and foraging 
name = 'arm'
for k = 1:length(fnames)
    load(fnames{k})
    
    if k == 1
        tmp = out;
    else
        tmp(:,k) = out;
    end
end

out = tmp;
%% load the corresponding arm data (load 1 at a time)
% load('...\Experiment4_data\cardBandit2arm.mat');
% load('...\Experiment_data\cardBandit3arm.mat');
% load('...\Experiment4_data\cardBandit4arm.mat');

modCol = {[0.6, 0.5, 0.1],[0.0, 0.6, 0.2],[0, 0, 0]};

%% plot the predicted p(switch) vs observed p(switch) for both rl and foraging. For individual sub-groups (i.e. only 2-armed, only 3-armed, only 4-armed)
nSs = 50 ; %47 for the 3armed; 50 for 2-armed and 4-armed. (because of NaNs).
nBoot = size(out,2);

% first, preallocate
[runLen,pSwitch,overThirty] = deal(NaN(nSs,nBoot,2));
c = 1;

% now do the human participants
for k = 1:length(trials)
    choices = [trials(k).trials.choice]; % yep

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

behOI = pSwitch;

mdlStr = {'rl','foraging'};

N_subjects_RMSE = size(behOI, 1); 

figure();
for plt = 1:2
    subplot(1,2,plt); axis square; hold on;
    tmp = squeeze(nanmean(behOI,2));
    plot(tmp(:,plt+1),tmp(:,1),'.',...
        'MarkerSize',15,'Color',modCol{plt})

 % 1. Calculate the TOTAL SSE distribution (1 x 100 vector)
    sse_dist = nansum((behOI(:,:,1) - behOI(:,:,plt+1)).^2, 1);
    
    % 2. Calculate the MEAN SQUARED ERROR (MSE) distribution
    mse_dist = sse_dist / N_subjects_RMSE;
    
    % 3. Calculate the ROOT MEAN SQUARED ERROR (RMSE) distribution
    rmse_dist = sqrt(mse_dist);
    
    set(gca,'FontSize',14)
    
    % Title now displays the Mean RMSE and 95% CI bounds
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
%% plot the distribution of the obs - pred

error_rl = tmp(:,2) - tmp(:,1);
error_foraging = tmp(:,3) - tmp(:,1);

figure;
set(gcf, 'Position', [100, 100, 300, 450]);
hold on;

%how big the bins are
BIN_WIDTH = 0.032;
X_MIN = -0.25;
X_MAX = 0.25;
bin_edges = X_MIN:BIN_WIDTH:X_MAX; 


h1 = histogram(error_rl, bin_edges, 'FaceColor', modCol{1}, 'EdgeColor', 'k');
h1.FaceAlpha = 0.7;


h2 = histogram(error_foraging, bin_edges, 'FaceColor', modCol{2}, 'EdgeColor', 'k');
h2.FaceAlpha = 0.7;

line([0 0], ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);

mean_err_rl = nanmean(error_rl);
mean_err_foraging = nanmean(error_foraging);


line([mean_err_rl mean_err_rl], ylim, 'Color', modCol{1}, 'LineStyle', '-', 'LineWidth', 3);
line([mean_err_foraging mean_err_foraging], ylim, 'Color', modCol{2}, 'LineStyle', '-', 'LineWidth', 3);

hold off;
 

title('Overlapping Distribution of Prediction Errors', 'FontSize', 16);
xlabel('Prediction Error (Predicted - Observed)', 'FontSize', 12);
ylabel('Frequency (Subjects)', 'FontSize', 12);

%legend
legend(sprintf('RL Model (mean=%.3f)', mean_err_rl), ...
       sprintf('Foraging Model (mean=%.3f)', mean_err_foraging), ...
       'Zero Error', ...
       'Location', 'best');
       
set(gca, 'FontSize', 11);
box off;


