%This code generates all stat panels for the first row of Figure 5
%(i.e. Fig 5 r,s,t) (AIC, switch probabilities as a function of predicted 
% probability of switch from simulations of the traditional RL model and 
% the foraging-RL model, and distribution of the difference in predicted 
% versus observed switch probability for both models).

% data, fits and simulations are on the figshare https://doi.org/10.6084/m9.figshare.32193990

%% look at correlation across params
% clear; clc;

% cd('...\Experiment5_fits');

% Load Eye Data
eye_data = load('fitRLtoModality2AB_20rounds_eye251111.mat');
eye_fits = eye_data.fits; 

% Load Touch Data
touch_data = load('fitRLtoModality2AB_20rounds_touchscreen251111.mat');
touch_fits = touch_data.fits;

%change these numbers if you want to look a different params !!!!!

% foraging (:,1) = alpha, (:,2) = beta, (:,3) = rho 
% rl (:,1) = alpha, (:,2) = beta

%..._fits(1) = rl 
%..._fits(2) = foraging 

params_forage_eye = eye_fits(1).params; 
params_forage_eye_alpha = params_forage_eye(:,3);

params_forage_touch = touch_fits(1).params; 
params_forage_touch_alpha = params_forage_touch(:,3);


[r_pearson, p_pearson] = corr(params_forage_touch_alpha, params_forage_eye_alpha, ...
                                'Type', 'Pearson', 'Rows', 'complete')


%% get AIC for the Experiment 5 (not in paper in the end)
clear;
% cd('...\Experiment5_fits');
load('fitRLtoModality2AB_20rounds_touchscreen251111.mat');

num_models = length(fits);
aic_values = NaN(num_models, 1);

for i = 1:2 %only need Foraging and RL
    k = fits(i).nParams;
    nll = fits(i).lik;
    aic = 2*k*29 + 2*nll;
    aic_values(i) = aic;
    fits(i).AIC = aic;
    modelname{i} = fits(i).modelName
end
figure;
set(gcf, 'Position', [100, 100, 200, 350]);
plot([2,1],aic_values(1:2), '-o', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0, 0, 0],  'MarkerFaceColor', [0, 0, 0],...
    'Color', [0, 0, 0]);
    xlim([0.75, 2 + 0.25]);
    xticks(1:aic_values(1:2));
    set(gca, 'XTickLabel', flip(modelname), 'FontSize', 11)



%% get AICs but depending on whether eye-tracking or touchscreen was presented first to participant

clear ; close all ; 
% cd('...\Experiment5_data')
% cd('...\Experiment5_fits');

%load fits
touchscreen = load('fitRLtoModality2AB_20rounds_touchscreen251111.mat');

eye =load('fitRLtoModality2AB_20rounds_eye251111.mat') ;

format = 'pdf';
mStr = {'foraging', 'RL'} ;
modCol = {[0.0, 0.6, 0.2],[0.6, 0.5, 0.1]};
paramStr= {'\alpha','\rho','\beta','\alpha','\beta'};

T = readtable('modality_participant_info.csv');
Tstruct = table2struct(T); 

num_subjects = size(eye.fits(1).params, 1);
subject_order = [Tstruct.eyetrackingFirst_1_Yes_];
if isrow(subject_order)
    subject_order = subject_order';
end
subject_order(subject_order == 0) = 2; 
assert(length(subject_order) == num_subjects, 'Subject count mismatch still exists!');

eye_first_idx = (subject_order == 1);
touch_first_idx = (subject_order == 2);

metrics_by_order = struct();
groups = {'eye_first', 'touch_first'};
group_indices = {eye_first_idx, touch_first_idx};
modalities_data = {eye, touchscreen};
modalities_names = {'eye', 'touchscreen'};
num_models_to_keep = 2; 
for g = 1:length(groups)
    current_group = groups{g};
    current_idx = group_indices{g};
    
    for m = 1:length(modalities_names)
        current_modality_name = modalities_names{m};
        fits = modalities_data{m}.fits;
        wonk = [fits.likelihood];
        wonk = wonk(:, 1:num_models_to_keep); 
        subgroup_wonk = wonk(current_idx, :);
        selex = sum(isnan(subgroup_wonk), 2) == 0;
        nSubjects_in_subgroup = sum(selex);
        metrics_by_order.(current_group).(current_modality_name).n = nSubjects_in_subgroup;
        if nSubjects_in_subgroup > 0
            logLik = nansum(subgroup_wonk(selex, :)); 
            nParams = [fits.nParams];
            nParams = nParams(1:num_models_to_keep); 
            [aic, bic] = aicbic(-logLik, nParams, nSubjects_in_subgroup); 
        else
            aic = nan(1, num_models_to_keep);
            bic = nan(1, num_models_to_keep);
        end
        metrics_by_order.(current_group).(current_modality_name).aic = aic;
        metrics_by_order.(current_group).(current_modality_name).logLik = logLik
        metrics_by_order.(current_group).(current_modality_name).bic = bic; 
    end
end

figure('Name', 'Modality Order Comparison (AIC)', 'Position', [150, 150, 600,450]);
sgtitle('model fit by modality order (AIC)', 'FontSize', 16, 'FontWeight', 'bold');
% now its 2 rows
aic_plot_data = [...
    metrics_by_order.eye_first.eye.aic;
    metrics_by_order.eye_first.touchscreen.aic;
    metrics_by_order.touch_first.eye.aic;
    metrics_by_order.touch_first.touchscreen.aic
]; % 4x2 matrix
n_values = [
    metrics_by_order.eye_first.eye.n;
    metrics_by_order.eye_first.touchscreen.n;
    metrics_by_order.touch_first.eye.n;
    metrics_by_order.touch_first.touchscreen.n
];

subplot_details = {
    {'Eye-First Group', 1:2, {'Eye Data', 'Touch Data'}}, ...
    {'Touch-First Group', 3:4, {'Eye Data', 'Touch Data'}}
};
modCol = {[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980]}; 
for p = 1:2 
    subplot(1, 2, p);
    hold on;
    
    details = subplot_details{p};
    subplot_title = details{1};
    data_rows = details{2};
    x_labels = details{3};
    
    current_aic_data = aic_plot_data(data_rows, :); % 2x2
    current_n_values = n_values(data_rows);
    
    x_locations = 1:2;
    x_points_all_models = [-0.1, 0.1]; 
    
    for i = 1:length(x_locations)
        x = x_locations(i);
        y_vals = current_aic_data(i, :); 
        x_points = x + x_points_all_models; 
        
        if ~any(isnan(y_vals))
            plot(x_points, y_vals, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
        end
        
        scatter(x_points(1), y_vals(1), 120, 'o', 'MarkerFaceColor', modCol{1}, 'MarkerEdgeColor', 'k', 'LineWidth', 1);
        scatter(x_points(2), y_vals(2), 120, 'o', 'MarkerFaceColor', modCol{2}, 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    end
    
    current_y_lims = ylim; 
    current_y_range = current_y_lims(2) - current_y_lims(1);

    text_y_position = current_y_lims(1) + current_y_range * 0.05; 
    
    for i = 1:length(x_locations)
        text(x_locations(i), text_y_position, sprintf('N = %d', current_n_values(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
    end

    title(subplot_title, 'FontSize', 14);
    ylabel('AIC (lower is better)');
    set(gca, 'XTick', x_locations, 'XTickLabel', x_labels, 'FontSize', 12);
    xlim([0.5, 2.5]);

    box off;
end
% legend
h1 = scatter(nan, nan, 120, 'o', 'MarkerFaceColor', modCol{1}, 'MarkerEdgeColor', 'k', 'LineWidth', 1);
h2 = scatter(nan, nan, 120, 'o', 'MarkerFaceColor', modCol{2}, 'MarkerEdgeColor', 'k', 'LineWidth', 1);
legend([h1, h2], {'Foraging Model', 'RL Model'}, 'Position', [0.4, 0.01, 0.2, 0.05], 'Orientation', 'horizontal', 'Box', 'off');

%% AIC WEIGHT CALCULATION (Model Selection Probability)

model_names = mStr(1:num_models_to_keep); % {'foraging', 'RL'}
fprintf('\n--- AIC WEIGHTS (Probability of being the best model) ---\n');

for g = 1:length(groups)
    current_group = groups{g};
    
    for m = 1:length(modalities_names)
        current_modality_name = modalities_names{m};
        

        group_title = sprintf('Group: %s, Data: %s', upper(current_group), upper(current_modality_name));
        fprintf('\n%s\n', group_title);
        fprintf('---------------------------------------------------\n');
        
        aic_values = metrics_by_order.(current_group).(current_modality_name).aic;
        n_subjects = metrics_by_order.(current_group).(current_modality_name).n;

        if n_subjects > 0 && ~any(isnan(aic_values))
            
            % minimum AIC value
            min_aic = min(aic_values);
            
            % Delta AIC (difference from the minimum)
            delta_aic = aic_values - min_aic;
            
%    Relative Likelihood (exp(-0.5 * Delta AIC))
            relative_likelihood = exp(-0.5 * delta_aic);
            
            sum_relative_likelihood = sum(relative_likelihood);
            aic_weights = relative_likelihood / sum_relative_likelihood;
            
            for i = 1:num_models_to_keep
                fprintf('Model: %-10s | AIC Weight: %.4f\n', model_names{i}, aic_weights(i));
            end
            metrics_by_order.(current_group).(current_modality_name).aic_weights = aic_weights; 
        else
            fprintf('Insufficient data (N=0) to calculate AIC Weights.\n');
        end
    end
end
fprintf('---------------------------------------------------\n');


%% we want to plot predicted p(switch) (for either eye-tracking or for touchscreen)
clear;
 % (load only 1)
% cd('...\simulationOutput_Experiment5\eye');
% cd('...\simulationOutput_Experiment5\touch');

files = dir;
fnames = {files(find(~cellfun(@isempty,strfind({files.name},'simResults')))).name};
nmod = 2; %only rl and foraging 
name = 'modality'
for k = 1:length(fnames)
    load(fnames{k})
    
    if k == 1
        tmp = out;
    else
        tmp(:,k) = out;
    end
end

out = tmp;
%% load data (load only 1)
% load('...\Experiment5_data\eyetrials.mat');
% load('...\Experiment5_data\touchtrials.mat');

modCol = {[0.6, 0.5, 0.1],[0.0, 0.6, 0.2],[0, 0, 0]};

%% for the p(switch), mean run length, and mean # runs > 30, we can plot the distribution of people
nSs = 29 ; 
nBoot = size(out,2);

% first, preallocate
[runLen,pSwitch,overThirty] = deal(NaN(nSs,nBoot,2));
c = 1;

% now do the human participants
for k = 1:length(trials)
    choices = [trials(k).choice]; % yep

    if length(unique(choices))>1
         
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

 % TOTAL SSE distribution (1 x 100 vector)
    sse_dist = nansum((behOI(:,:,1) - behOI(:,:,plt+1)).^2, 1);
    
    % MEAN SQUARED ERROR (MSE) distribution
    mse_dist = sse_dist / N_subjects_RMSE;
    
    %  ROOT MEAN SQUARED ERROR (RMSE) distribution
    rmse_dist = sqrt(mse_dist);
    
    set(gca,'FontSize',14)
    
    title(sprintf('Model: %s\nMean RMSE = %2.4f (95%% CI: [%2.4f, %2.4f])',...
        mdlStr{plt}, nanmean(rmse_dist), quantile(rmse_dist, 0.025), quantile(rmse_dist, 0.975)))
    
    tmpPLOT = [xlim,ylim];
    xlim([0 max(tmpPLOT)]); ylim([0 max(tmpPLOT)]);
    line([0 max(tmpPLOT)],[0 max(tmpPLOT)],'Color','k')
    xlabel(strcat(mdlStr{plt},' predicted p(switch)'))
    ylabel(strcat('observed p(switch)'))
        xlim([0 1]);
        ylim([0 1]);
end

%% now plot the distribution

error_rl = tmp(:,2) - tmp(:,1);
error_foraging = tmp(:,3) - tmp(:,1);

figure;
set(gcf, 'Position', [100, 100, 500, 450]);
hold on;

BIN_WIDTH = 0.02;
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


legend(sprintf('RL Model (mean=%.3f)', mean_err_rl), ...
       sprintf('Foraging Model (mean=%.3f)', mean_err_foraging), ...
       'Zero Error', ...
       'Location', 'best');
       
set(gca, 'FontSize', 11);
box off;

%% stats for dist of obs - pred


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


%% foraging index
clear; clc;

% cd('...\Experiment5_fits');

% Load Eye Data
eye_data = load('fitRLtoModality2AB_20rounds_eye251111.mat');
eye_fits = eye_data.fits; 

% Load Touch Data
touch_data = load('fitRLtoModality2AB_20rounds_touchscreen251111.mat');
touch_fits = touch_data.fits;

L_forage_eye = eye_fits(1).likelihood(:); 
L_rl_eye     = eye_fits(2).likelihood(:);

L_rl_eye = - L_rl_eye;
L_forage_eye = -L_forage_eye;

% for. index
eye_foraging_index = L_forage_eye - L_rl_eye;


L_forage_touch = touch_fits(1).likelihood(:);
L_rl_touch     = touch_fits(2).likelihood(:);

L_forage_touch = -L_forage_touch
L_rl_touch = -L_rl_touch

touch_foraging_index = L_forage_touch - L_rl_touch;

[r_pearson, p_pearson] = corr(eye_foraging_index, touch_foraging_index, ...
                                'Type', 'Pearson', 'Rows', 'complete');

fprintf('\n--- Correlation Results ---\n');
fprintf('Spearman rho: %.3f (p = %.4f)\n', r_pearson, p_pearson);


figure('Name', 'Stability of Foraging Index');
scatter(eye_foraging_index, touch_foraging_index, 70, 'filled', 'MarkerFaceColor', 'b');
hold on;

h = lsline;
set(h, 'LineWidth', 2, 'Color', 'r');

% Formatting
xlabel('Eye Modality (-LLFOR - -LLRL)');
ylabel('Touch Modality (-LLFOR - -LLRL)');
grid on;

% Add reference lines at 0 to show the quadrants
xline(0, '--k', 'Alpha', 0.5);
yline(0, '--k', 'Alpha', 0.5);


