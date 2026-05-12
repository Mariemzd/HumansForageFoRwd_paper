%This code generates all stat panels for the second row of Figure 5
%(i.e. Fig 5 H,I,J) (Autocorrelation, AIC, switch probabilities as a function of predicted 
% probability of switch from simulations of the traditional RL model and 
% the foraging-RL model, and distribution of the difference in predicted 
% versus observed switch probability for both models).

% data and simulations are on the figshare https://doi.org/10.6084/m9.figshare.32193990

%% Autocorrelation stuff (Figure 5 H)

% cd('...\Experiment3_data');

% set some free parameters
halfThreshold = 0.75;
nTrials = 3;
removeOutliers = true;
outThresh = 80;
maxLag = 100;
nbins = 10;

% load the files
files = dir;
rights = ~cellfun(@isempty,strfind({files.name},'volatilityLMH_newstates2025'));
mats = ~cellfun(@isempty,strfind({files.name},'.mat'));
matfiles = {files(and(rights,mats)).name};

% preallocate
k = 1; [halfLife,decayAfter,pExplore,color] = deal(NaN(270,1));
colorMap = [0.521 0.709 0.541; 0.901 0.760 0.431; 0.701 0.552 0.737];

% store ACFs as cell arrays to handle unequal session lengths
all_acfs   = {};
all_acfs_1 = {};
all_acfs_2 = {};
all_acfs_3 = {};

% loop through datasets, then subjects
for dB = 1%:length(matfiles)
    load(matfiles{dB})
    for subj = 1:length(trials)

        correct = [trials(subj).trials.practice] == 0;
        t = trials(subj).trials(correct);
        rwds = vertcat(t.reward_seed);

        if sum(isnan(rwds(end,:))) > 1
            rwds(end,:) = rwds(end-1,:);
        end

        % calculate the average autocorrelation for each arm
        acf = NaN(3, size(rwds,1)*2-1);
        for ch = 1:3
            acf(ch,:) = xcorr(rwds(:,ch),'coeff')';
        end
        acf = nanmean(acf, 1);

        % store in cell arrays
        all_acfs{end+1} = acf;

        switch trials(subj).color
            case 1
                all_acfs_1{end+1} = acf;
            case 2
                all_acfs_2{end+1} = acf;
            case 3
                all_acfs_3{end+1} = acf;
        end

        if halfLife(k) > 80
            keyboard()
        end
    end
end

% pad with NaN and convert to matrices
% all subjects
max_len = max(cellfun(@length, all_acfs));
all_acfs_mat = NaN(length(all_acfs), max_len);
for i = 1:length(all_acfs)
    n = length(all_acfs{i});
    pad = floor((max_len - n) / 2);
    all_acfs_mat(i, pad+1:pad+n) = all_acfs{i};
end

% group 1
max_len = max(cellfun(@length, all_acfs_1));
all_acfs_mat_1 = NaN(length(all_acfs_1), max_len);
for i = 1:length(all_acfs_1)
    n = length(all_acfs_1{i});
    pad = floor((max_len - n) / 2);
    all_acfs_mat_1(i, pad+1:pad+n) = all_acfs_1{i};
end

% group 2
max_len = max(cellfun(@length, all_acfs_2));
all_acfs_mat_2 = NaN(length(all_acfs_2), max_len);
for i = 1:length(all_acfs_2)
    n = length(all_acfs_2{i});
    pad = floor((max_len - n) / 2);
    all_acfs_mat_2(i, pad+1:pad+n) = all_acfs_2{i};
end

% group 3
max_len = max(cellfun(@length, all_acfs_3));
all_acfs_mat_3 = NaN(length(all_acfs_3), max_len);
for i = 1:length(all_acfs_3)
    n = length(all_acfs_3{i});
    pad = floor((max_len - n) / 2);
    all_acfs_mat_3(i, pad+1:pad+n) = all_acfs_3{i};
end

% Calculate the averages and standard deviations for each category
avg_acf_1 = nanmean(all_acfs_mat_1, 1);
avg_acf_2 = nanmean(all_acfs_mat_2, 1);
avg_acf_3 = nanmean(all_acfs_mat_3, 1);

std_acf_1 = nanstd(all_acfs_mat_1, 0, 1);
std_acf_2 = nanstd(all_acfs_mat_2, 0, 1);
std_acf_3 = nanstd(all_acfs_mat_3, 0, 1);

% Define the colorMap
colorMap = [0.521 0.709 0.541; 0.901 0.760 0.431; 0.701 0.552 0.737];

% Plot the average ACFs with error bars
figure('Position',[440   214   400   450]); hold on;
for i = 1:3
    % Select the average and std deviation based on category
    if i == 1
        avg_acf = avg_acf_1;
        std_acf = std_acf_1;
    elseif i == 2
        avg_acf = avg_acf_2;
        std_acf = std_acf_2;
    else
        avg_acf = avg_acf_3;
        std_acf = std_acf_3;
    end
acf_x = -(size(rwds,1)-1):(size(rwds,1)-1);
    % Plot the average ACF with shaded error region
    ph = plot(acf_x, avg_acf, 'Color', colorMap(i, :)); % Plot the average ACF
    fill([acf_x, fliplr(acf_x)], [avg_acf+std_acf, fliplr(avg_acf-std_acf)],...
        colorMap(i, :), 'FaceAlpha', 0.5, 'LineStyle', 'none');
end

% Adjust plot settings
xlim([-maxLag, maxLag]);
ylim([0.3, 1]);
set(gca, 'FontSize', 16);
xlabel('lag (trials)');
ylabel('correlation');
hold on;
%
%now plot the Experiment 1 autocorrelation on top

% cd('...'); %%%%%%%%%%%%%%%%%%%% figshare directory for experiment 1 %%%%%%%%%%%%%%%%

% set some free parameters
halfThreshold = 0.75;
nTrials = 3;
removeOutliers = true;
outThresh = 80;
maxLag = 100;
nbins = 10;

% load the files
files = dir;
rights = ~cellfun(@isempty,strfind({files.name},'singleiti_202203011023_lightweight'));
mats = ~cellfun(@isempty,strfind({files.name},'.mat'));
matfiles = {files(and(rights,mats)).name};

% preallocate
k = 1; [halfLife,decayAfter,pExplore,color] = deal(NaN(258,1));
colorMap = [0.221 0.409 0.541];

% store ACFs as cell arrays to handle unequal session lengths
all_acfs   = {};
all_acfs_1 = {};

% loop through datasets, then subjects
for dB = 1%:length(matfiles)
    load(matfiles{dB})
    for subj = 1:length(trials)

        correct = [trials(subj).trials.practice] == 0;
        t = trials(subj).trials(correct);
        rwds = vertcat(t.reward_seed);

        if sum(isnan(rwds(end,:))) > 1
            rwds(end,:) = rwds(end-1,:);
        end

        % calculate the average autocorrelation for each arm
        acf = NaN(3, size(rwds,1)*2-1);
        for ch = 1:2
            acf(ch,:) = xcorr(rwds(:,ch),'coeff')';
        end
        acf = nanmean(acf, 1);

        % store in cell arrays
        all_acfs{end+1} = acf;

        if halfLife(k) > 80
            keyboard()
        end
    end
end

% pad with NaN and convert to matrices
% all subjects
max_len = max(cellfun(@length, all_acfs));
all_acfs_mat = NaN(length(all_acfs), max_len);
for i = 1:length(all_acfs)
    n = length(all_acfs{i});
    pad = floor((max_len - n) / 2);
    all_acfs_mat(i, pad+1:pad+n) = all_acfs{i};
end

% Calculate the averages and standard deviations for each category
avg_acf_1 = nanmean(all_acfs_mat, 1);

std_acf_1 = nanstd(all_acfs_mat, 0, 1);

% Plot the average ACFs with error bars
% figure('Position',[440   214   400   450]); hold on;
for i = 1
        avg_acf = avg_acf_1;
        std_acf = std_acf_1;

 end
acf_x = -(size(rwds,1)-1):(size(rwds,1)-1);
    % Plot the average ACF with shaded error region
    ph = plot(acf_x, avg_acf, 'Color', colorMap(i, :)); % Plot the average ACF
    fill([acf_x, fliplr(acf_x)], [avg_acf+std_acf, fliplr(avg_acf-std_acf)],...
        colorMap(i, :), 'FaceAlpha', 0.5, 'LineStyle', 'none');

% Adjust plot settings
xlim([-maxLag, maxLag]);
ylim([0.3, 1]);
set(gca, 'FontSize', 16);
xlabel('lag (trials)');
ylabel('correlation');
hold on;


%% get AIC for the Experiment 3
clear; clc;

% cd('...\Experiment3_data');
% addpath('...\Experiment3_fits');

load('fitRLtoVolatilityMTurk_20rounds_251106.mat');
load('volatilityLMH_newstates2025.mat');

num_models = length(fits);
aic_values = NaN(num_models, 1);

%get the IDs out
subject_group_assignments = [trials.color];
groups = unique(subject_group_assignments);
num_groups = length(groups);
group_subject_counts = [90, 91, 88];

for i = 1:2 %only need Foraging and RL
    k = fits(i).nParams;
    nll_all = fits(i).likelihood;
    modelname{i} = fits(i).modelName

    %Seperate the AIC by the 3 groups
    for g_idx = 1:num_groups
        current_group_id = groups(g_idx);
        num_subjects_for_this_group = group_subject_counts(g_idx);
        group_indices = (subject_group_assignments == current_group_id);
        nlls_for_this_group = nll_all(group_indices);
        total_group_nll = nansum(nlls_for_this_group);
        aic = 2 * k * num_subjects_for_this_group + 2 * total_group_nll;

        % Store the results in our structure
        group_field_name = ['group_' num2str(current_group_id)];
        nll_plot.(group_field_name).(modelname{i}).total_group_nll = total_group_nll;
        aic_results.(group_field_name).(modelname{i}).aic = aic;
        aic_results.(group_field_name).(modelname{i}).total_nll = total_group_nll;
        aic_results.(group_field_name).(modelname{i}).k = k;
        aic_results.(group_field_name).(modelname{i}).num_subjects = sum(group_indices);
    end
end

%name the data formatted for the figure
num_models = 2;
aic_plot_data = NaN(num_groups, num_models);

for g = 1:3 %3 groups
    current_group_id = groups(g);
    group_field_name = ['group_' num2str(current_group_id)];

    for m = 1:num_models
        current_model_name = modelname{m};

        %      Model 1 (Foraging) | Model 2 (RL)
        % G1 | aic_g1_m1          | aic_g1_m2
        % G2 | aic_g2_m1          | aic_g2_m2
        % G3 | aic_g3_m1          | aic_g3_m2
        aic_plot_data(g, m) = aic_results.(group_field_name).(current_model_name).aic;
    end
end

figure;
set(gcf, 'Position', [100, 100, 200, 350]);
hold on;
colors = [0, 0, 1; 1, 0, 0; 0, 0.5, 0];
legend_labels = cell(num_groups, 1);

for g = 1:num_groups
    group_aic_data = aic_plot_data(g, :);
    plot([2, 1], group_aic_data, '-o', 'Color', colors(g, :), 'MarkerFaceColor', colors(g, :),'LineWidth', 2,'MarkerSize', 10);
end
hold off;
xlim([0.75, 2 + 0.25]);
xticks(1:aic_values(1:2));

%% Calculate AIC Weights

aic_weights_by_group = NaN(num_groups, 2); 
aic_weights_results = struct();


for g = 1:num_groups % Iterate through the 3 groups
    
    current_group_id = groups(g);
    group_field_name = ['group_' num2str(current_group_id)];
    group_aic_values = aic_plot_data(g, :); 
    min_aic = min(group_aic_values); 
    delta_aic = group_aic_values - min_aic;
    relative_likelihood = exp(-0.5 * delta_aic);
    sum_relative_likelihood = sum(relative_likelihood);
    aic_weights = relative_likelihood / sum_relative_likelihood;
    

    aic_weights_by_group(g, :) = aic_weights;
    aic_weights_results.(group_field_name).Foraging_Weight = aic_weights(1);
    aic_weights_results.(group_field_name).RL_Weight = aic_weights(2);
    
    fprintf('Group %d \n', current_group_id);
    fprintf('  Foraging Model Weight: %.4f\n', aic_weights(1));
    fprintf('  RL Model Weight:  %.4f\n', aic_weights(2));
end

%% individual comparison for all dataset 

load('fitRLtoVolatilityMTurk_20rounds_251106.mat');

aicF= 2*3 + 2*fits(1).likelihood ;
aicR= 2*2 + 2*fits(2).likelihood ;
% I use AIC here and not loglikelihood as suggested in this paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC2703732/#S2
% p = signrank(aicF,aicRence = AIC) 

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
df_equivalent = n - 1

mean_diff = nanmean(differences)
diff_range = [min(differences), max(differences)]


%% get Median for individual comparision

median_stats = struct();

for g_idx = 1:num_groups
    current_group_id = groups(g_idx);
    group_field_name = ['group_' num2str(current_group_id)];
    group_indices = (subject_group_assignments == current_group_id);
    for i = 1:2 % Iterate over Foraging (1) and RL (2)
        model_name_str = modelname{i};
        k = fits(i).nParams;
        nll_all = fits(i).likelihood; % Individual NLLs
        nlls_for_this_group = nll_all(group_indices);
        individual_aics_for_this_group = 2 * k + 2 * nlls_for_this_group;
        
        median_nll_individual = nanmedian(nlls_for_this_group);
        median_aic_individual = nanmedian(individual_aics_for_this_group);
        
        % Store the results in the new structure
        median_stats.(group_field_name).(model_name_str).median_nll = median_nll_individual;
        median_stats.(group_field_name).(model_name_str).median_aic = median_aic_individual;
    end
end

%% individual differences
kF = fits(1).nParams; % Should be 3 for Foraging model
kR = fits(2).nParams; % Should be 2 for RL model


aicF_all = 2 * kF + 2 * fits(1).likelihood;
aicR_all = 2 * kR + 2 * fits(2).likelihood;
group_signrank_results = struct();

for g_idx = 1:num_groups
    current_group_id = groups(g_idx);
    group_field_name = ['group_' num2str(current_group_id)];
    
    group_indices = (subject_group_assignments == current_group_id);
    
    aicF_group = aicF_all(group_indices);
    aicR_group = aicR_all(group_indices);
    
    nan_indices = isnan(aicF_group) | isnan(aicR_group);


aicF_valid = aicF_group(~nan_indices);
aicR_valid = aicR_group(~nan_indices);


[p, h, stats] = signrank(aicF_valid, aicR_valid);

N_valid = length(aicF_valid); % this is 90 for low vol.
    

    median_diff = nanmedian(aicF_group - aicR_group);

    group_signrank_results.(group_field_name).p_value = p;
    group_signrank_results.(group_field_name).median_diff_AIC_F_minus_R = median_diff;


    fprintf('Group %d (N=%d):\n', current_group_id);
    fprintf('  P-value (AIC_F vs AIC_R): p = %.5f\n', p);
end



%% Plot predicted p(switch) for ALL groups
clear; clc;

% cd('...\Experiment3_data');
load('volatilityLMH_newstates2025.mat');

num_groups = 7;
% base_sim_path = ...\simulationOutputs_Experiment3';
all_trials = trials; 
clear trials; 

mdlStr = {'rl','foraging'};

modCol = {[0.6, 0.5, 0.1],[0.0, 0.6, 0.2]}; 

group_colors = lines(num_groups); 

Aggregated_behOI_4D = []; 

all_results = cell(num_groups, 1);

for g = 1:num_groups
    fprintf('\n--- Processing Group %d ---\n', g);
    
    group_folder = sprintf('vGroup%d', g);
    group_path = fullfile(base_sim_path, group_folder);
    
    fprintf('Loading simulation files from: %s\n', group_path);
    cd(group_path);
    
    files = dir;
    fnames = {files(find(~cellfun(@isempty,strfind({files.name},'simResults')))).name};
    
    if isempty(fnames)
        warning('No simResults files found in %s. Skipping group.', group_path);
        continue;
    end
    
    for k = 1:length(fnames)
        load(fnames{k})
        if k == 1
            tmp_out = out;
        else
            tmp_out(:,k) = out;
        end
    end
    out = tmp_out;
    nBoot = size(out,2);
    fprintf('Loaded %d simulation boots.\n', nBoot);


    trials = all_trials([all_trials.group] == g);
    nSs = length(trials); % Dynamic nSs
    
    if nSs == 0
        warning('No subjects found for group %d. Skipping.', g);
        continue;
    end
    fprintf('Found %d subjects for Group %d.\n', nSs, g);


    [runLen,pSwitch,overThirty] = deal(NaN(nSs,nBoot,2));
    c = 1; % This will count *valid* subjects
    for k = 1:length(trials)
        selex = [trials(k).trials.practice]==0;
        choices = [trials(k).trials(selex).choice];
        if length(unique(choices))>1
            pSwitch(c,:,1) = deal(nanmean([diff(choices)~=0]));
            runLen(c,:,1) = deal(nanmean([diff(find(diff(choices)~=0))]));
            overThirty(c,:,1) = deal(nanmean([diff(find(diff(choices)~=0))]>30));
            c = c+1;
        end
    end

    valid_subject_count = c - 1;
    
    if valid_subject_count == 0
        warning('No valid subjects (with >1 choice) found for group %d. Skipping.', g);
        continue; 
    end
    
    if valid_subject_count < nSs
        fprintf('Trimming matrices from %d to %d subjects to account for exclusions.\n', nSs, valid_subject_count);
        pSwitch = pSwitch(1:valid_subject_count, :, :);
        runLen = runLen(1:valid_subject_count, :, :);
        overThirty = overThirty(1:valid_subject_count, :, :);
    end

    for model = 1:2
        switch model
            case 1
                indx = find(strcmp({out(:,1).type},'rl'));
            case 2
                indx = find(strcmp({out(:,1).type},'f'));
        end
        
        if isempty(indx)
            warning('Could not find model type for model %d in group %d', model, g);
            continue;
        end
        
        for boot = 1:nBoot
            choices = {out(indx,boot).choices};
            
            % Check for size mismatch *before* assignment
            n_model_subjects = length(arrayfun(@(k) nanmean(diff(choices{k})~=0),1:length(choices))');
            if n_model_subjects ~= valid_subject_count
                error('CRITICAL ERROR in Group %d: Human valid subjects (%d) does not match model subjects (%d).', ...
                      g, valid_subject_count, n_model_subjects);
            end
            
            % This assignment should now work
            pSwitch(:,boot,model+1) = arrayfun(@(k) nanmean(diff(choices{k})~=0),1:length(choices))';
            runLen(:,boot,model+1) = arrayfun(@(k) nanmean([diff(find(diff(choices{k})~=0))]),1:length(choices))';
            overThirty(:,boot,model+1) = arrayfun(@(k) nanmean([diff(find(diff(choices{k})~=0))]>30),1:length(choices))';
        end
    end
    
    % store Results
    behOI = pSwitch;
    % tmp is [valid_subject_count x 3] -> [Observed, Pred_RL, Pred_Foraging]
    tmp_results = squeeze(nanmean(behOI,2));
    all_results{g} = tmp_results;
        if isempty(Aggregated_behOI_4D)
        Aggregated_behOI_4D = pSwitch;
    else
        % Concatenate along the subject dimension (Dimension 1)
        Aggregated_behOI_4D = cat(1, Aggregated_behOI_4D, pSwitch);
    end

end
fprintf('\n--- All Groups Processed ---\n');
disp('Stats saved in ''stats_results'' structure.');

%% now plot the pred obs plot

% The full subject-boot-model data (Total_Subjects x NBoot x 3)
behOI_full = Aggregated_behOI_4D; 

combined_array = squeeze(nanmean(behOI_full, 2)); 

figure('Position', [100, 100, 1000, 450]);

N_subjects = 269; 

for plt = 1:2 % 1=RL (column 2), 2=Foraging (column 3)
    subplot(1,2,plt);
    axis square;
    hold on;
    

    tmp = combined_array; 
    observed = tmp(:, 1);
    predicted = tmp(:, plt+1);

    % Plot individual subject means
    plot(predicted, observed, '.', 'MarkerSize', 15, 'Color', modCol{plt})
    

    sse_dist = nansum((behOI_full(:,:,1) - behOI_full(:,:,plt+1)).^2, 1);

    mse_dist = sse_dist / N_subjects;

    rmse_dist = sqrt(mse_dist);

    title(sprintf('Model: %s\nMean RMSE = %2.4f (95%% CI: [%2.4f, %2.4f])',...
        mdlStr{plt}, nanmean(rmse_dist), quantile(rmse_dist, 0.025), quantile(rmse_dist, 0.975)))
    
    set(gca, 'FontSize', 14)
    
    line([0 0.8],[0 0.8],'Color','k', 'LineStyle', '--')
    
    xlabel(strcat(mdlStr{plt},' predicted p(switch)'))
    ylabel('observed p(switch)')
end
sgtitle('Observed vs Predicted p(switch) Across All Groups', 'FontSize', 16);


%% Non-Parametric Statistical Tests (Wilcoxon Signed-Rank Test)

tmp = squeeze(nanmean(behOI_full,2)); % pull the behavior back in
sqErr_RL = (tmp(:,1) - tmp(:,2)).^2;
sqErr_F = (tmp(:,1) - tmp(:,3)).^2;
err_RL = (tmp(:,2) - tmp(:,1));
err_F = (tmp(:,3) - tmp(:,1));

disp('--- Wilcoxon Signed-Rank Test Results ---');
[p_sqErr, h_sqErr, stats_sqErr] = signrank(sqErr_RL, sqErr_F);

z_score = stats_sqErr.zval
W = stats_sqErr.signedrank
p_sqErr

differences = sqErr_RL - sqErr_F;
n = sum(~isnan(differences) & differences ~= 0);
df_equivalent = n - 1

mean_diff = nanmean(differences)
diff_range = [min(differences), max(differences)]

%% distribution

all_errors_rl = [];
all_errors_foraging = [];
for g = 1:num_groups
    tmp = all_results{g};
    if ~isempty(tmp)
        % tmp(:,1) = Observed
        % tmp(:,2) = RL
        % tmp(:,3) = Foraging
        group_errors_rl = tmp(:,2) - tmp(:,1);
        group_errors_foraging = tmp(:,3) - tmp(:,1);
        
        % Append this group's errors to the master list
        all_errors_rl = [all_errors_rl; group_errors_rl];
        all_errors_foraging = [all_errors_foraging; group_errors_foraging];
    end
end
fprintf('Generating combined error histogram for %d total subjects...\n', length(all_errors_rl));

figure;
set(gcf, 'Position', [100, 100, 600, 450]);
hold on;

BIN_WIDTH = 0.03;
X_MIN = -0.5;
X_MAX = 0.5;
bin_edges = X_MIN:BIN_WIDTH:X_MAX; 

h1 = histogram(all_errors_rl, bin_edges, 'FaceColor', modCol{1}, 'EdgeColor', 'k');
h1.FaceAlpha = 0.7;


h2 = histogram(all_errors_foraging, bin_edges, 'FaceColor', modCol{2}, 'EdgeColor', 'k');
h2.FaceAlpha = 0.7;

yLims = ylim; 


line([0 0], yLims, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2); 

mean_err_rl = nanmean(all_errors_rl);
mean_err_foraging = nanmean(all_errors_foraging);
line([mean_err_rl mean_err_rl], yLims, 'Color', modCol{1}, 'LineStyle', '-', 'LineWidth', 3);
line([mean_err_foraging mean_err_foraging], yLims, 'Color', modCol{2}, 'LineStyle', '-', 'LineWidth', 3);
hold off;

set(gca, 'YAxisLocation', 'origin');

title('Overlapping Distribution of Prediction Errors', 'FontSize', 16);
xlabel('Prediction Error (Predicted - Observed)', 'FontSize', 12);
ylabel('Frequency (Subjects)', 'FontSize', 12);

%legend
legend(sprintf('RL Model (n=%d, mean=%.3f)', length(all_errors_rl), mean_err_rl), ...
       sprintf('Foraging Model (n=%d, mean=%.3f)', length(all_errors_foraging), mean_err_foraging), ...
       'Zero Error', ...
       'Location', 'best');
       
set(gca, 'FontSize', 11);

xlim([X_MIN X_MAX]); 
box off;
disp('All plots generated.');