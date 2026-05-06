
% figure 1F-G && supplementary figure S1A-D

% Create a GLM to explicitly test a central assumption of the compare-to-threshold model,
% which predicts that choice repetition depends only on the value of the chosen option (VRepeat),
% and *not* on the value of the unchosen option (VSwitch).

% Create and compare three GLMs to test assumptions of decision models.
% Model 1 (Full): RepeatChoice ~ VRepeat + VSwitch
% Model 2 (VRepeat Only): RepeatChoice ~ VRepeat
% Model 3 (VSwitch Only): RepeatChoice ~ VSwitch

% VJL2026

%% directories
clear; close all ;

datafile =  '/data/singleiti_202203011023_lightweight.mat' ; 
load(datafile)

%%
use_subj_vals = 0 ; % 0 = objective probabilities

% For the subjective values,you can either have static bandit Baysian update or diffusing bandit Bayesian update.
static = 1 ; % 0 = diffusing bandit bayesian update

nParticipants = numel(trials);
skipped_participants = [];

% For the full model analysis
pVsVSwitch_full = nan(nParticipants,1);
pVsVRepeat_full = nan(nParticipants,1);
betasVSwitch = nan(nParticipants,1);
betasVRepeat = nan(nParticipants,1);

% For the simple model p-value analysis
pVsVRepeat_VRepeatOnlyModel = nan(nParticipants, 1);
pVsVSwitch_VSwitchOnlyModel = nan(nParticipants, 1);

% quick reminder
% Col 1: Full Model, Col 2: VRepeat Only, Col 3: VSwitch Only
AICs = nan(nParticipants, 3);
BICs = nan(nParticipants, 3);
LogLikelihoods = nan(nParticipants, 3);

% Loop over participants and fit GLM
for participant = 1:nParticipants

    selex = [trials(participant).trials.practice]==0 ;
    good = trials(participant).trials(selex); %don't keep practice trials

    choices = [good.choice];
    reward = [good.reward];

    if use_subj_vals
        reward_vals = BayesianIdealObserver(choices+1, reward,static) ; % subjective values instead
        reward_vals = reward_vals' ;
    else
        reward_vals = vertcat(good.reward_seed) ;
    end

    L = length(choices);

    % Skip participants who always choose one option
    unique_choices = unique(choices);
    if numel(unique_choices) == 1
        skipped_participants(end+1) = participant;
        fprintf('Skipping participant %d\n', participant);
        continue;
    end

    % Preallocate
    repeat_choice = nan(1, L);
    V_repeat      = nan(1, L);
    V_switch      = nan(1, L);

    % Fill in from trial 2 to L
    for i = 2:L
        % did they repeat? (1 = same as previous, 0 = switched)
        repeat_choice(i) = (choices(i) == choices(i-1));

        % Get value of chosen and unchosen options based on previous choice
        prev_choice = choices(i-1);

        if prev_choice == 0
            V_repeat(i) = reward_vals(i, 1);  % if they chose “0” last time, that option’s prob
            V_switch(i) = reward_vals(i, 2);  % the other arm’s prob
        else
            V_repeat(i) = reward_vals(i, 2);
            V_switch(i) = reward_vals(i, 1);
        end
    end

    % table and fit logistic regression
    valid_idx = 2:L; % Exclude trial 1 from the regression
    tbl = table( ...
        repeat_choice(valid_idx)', ...
        V_repeat(valid_idx)', ...
        V_switch(valid_idx)', ...
        'VariableNames', {'RepeatChoice','VRepeat','VSwitch'} );
    try

        % Model 1: Full Model
        mdl_full = fitglm(tbl, 'RepeatChoice ~ VRepeat + VSwitch', ...
            'Distribution','binomial','Link','logit', ...
            'LikelihoodPenalty','jeffreys-prior');

        % Model 2: VRepeat Only
        mdl_vrepeat = fitglm(tbl, 'RepeatChoice ~ VRepeat', ...
            'Distribution','binomial','Link','logit', ...
            'LikelihoodPenalty','jeffreys-prior');

        % Model 3: VSwitch Only
        mdl_vswitch = fitglm(tbl, 'RepeatChoice ~ VSwitch', ...
            'Distribution','binomial','Link','logit', ...
            'LikelihoodPenalty','jeffreys-prior');

        % store everything

        % Extract p-values and betas from the full model
        idxVs = strcmp(mdl_full.Coefficients.Properties.RowNames, 'VSwitch');
        pVsVSwitch_full(participant) = mdl_full.Coefficients.pValue(idxVs);
        betasVSwitch(participant) = mdl_full.Coefficients.Estimate(idxVs);

        idxVr = strcmp(mdl_full.Coefficients.Properties.RowNames, 'VRepeat');
        pVsVRepeat_full(participant) = mdl_full.Coefficients.pValue(idxVr);
        betasVRepeat(participant) = mdl_full.Coefficients.Estimate(idxVr);

        %Extract p-values from the simpler models

        % From Model 2 (VRepeat Only)
        idxVr_m2 = strcmp(mdl_vrepeat.Coefficients.Properties.RowNames, 'VRepeat');
        pVsVRepeat_VRepeatOnlyModel(participant) = mdl_vrepeat.Coefficients.pValue(idxVr_m2);

        % From Model 3 (VSwitch Only)
        idxVs_m3 = strcmp(mdl_vswitch.Coefficients.Properties.RowNames, 'VSwitch');
        pVsVSwitch_VSwitchOnlyModel(participant) = mdl_vswitch.Coefficients.pValue(idxVs_m3);

        % Store AIC and BIC for model comparison
        AICs(participant, 1) = mdl_full.ModelCriterion.AIC;
        AICs(participant, 2) = mdl_vrepeat.ModelCriterion.AIC;
        AICs(participant, 3) = mdl_vswitch.ModelCriterion.AIC;

        BICs(participant, 1) = mdl_full.ModelCriterion.BIC;
        BICs(participant, 2) = mdl_vrepeat.ModelCriterion.BIC;
        BICs(participant, 3) = mdl_vswitch.ModelCriterion.BIC;

        LogLikelihoods(participant, 1) = mdl_full.LogLikelihood;
        LogLikelihoods(participant, 2) = mdl_vrepeat.LogLikelihood;
        LogLikelihoods(participant, 3) = mdl_vswitch.LogLikelihood;

    catch ME
        fprintf('Error fitting models for participant %d: %s\n', participant, ME.message);
    end
end

% other stuff
% Summary of issues
fprintf('\n=== Summary ===\n');
fprintf('Skipped participants (only chose one option): %s\n', mat2str(skipped_participants));
fprintf('Total participants analyzed: %d\n', nParticipants - numel(skipped_participants));


% Analysis of the full model's coefficients
fprintf('\n=== Full Model (VRepeat + VSwitch) Coefficient Analysis ===\n');
alpha = 0.05;

% correcting for multiple comparisons
correctionThresholds_VRepeat_full = HolmBonferroni(pVsVRepeat_full, alpha);
correctionThresholds_VSwitch_full = HolmBonferroni(pVsVSwitch_full, alpha);
significant_VRepeat_full = pVsVRepeat_full < correctionThresholds_VRepeat_full;
significant_VSwitch_full = pVsVSwitch_full < correctionThresholds_VSwitch_full;
fprintf('Sign. part. after correction: VRepeat = %d/%d, VSwitch = %d/%d\n', ...
    sum(significant_VRepeat_full, 'omitnan'), numel(pVsVRepeat_full)-sum(isnan(pVsVRepeat_full)), ...
    sum(significant_VSwitch_full, 'omitnan'), numel(pVsVSwitch_full)-sum(isnan(pVsVSwitch_full)));


%%% analysis of the VRepeat Only model's coefficients
fprintf('\n=== VRepeat Only Model Coefficient Analysis ===\n');
correctionThresholds_VRepeat_m2 = HolmBonferroni(pVsVRepeat_VRepeatOnlyModel, alpha);
significant_VRepeat_m2 = pVsVRepeat_VRepeatOnlyModel < correctionThresholds_VRepeat_m2;
fprintf('Sign. part. after correction: VRepeat = %d/%d\n', ...
    sum(significant_VRepeat_m2, 'omitnan'), numel(pVsVRepeat_VRepeatOnlyModel)-sum(isnan(pVsVRepeat_VRepeatOnlyModel)));


%%% analysis of the VSwitch Only model's coefficients
fprintf('\n=== VSwitch Only Model Coefficient Analysis ===\n');
correctionThresholds_VSwitch_m3 = HolmBonferroni(pVsVSwitch_VSwitchOnlyModel, alpha);
significant_VSwitch_m3 = pVsVSwitch_VSwitchOnlyModel < correctionThresholds_VSwitch_m3;
fprintf('Sign. part. after correction: VSwitch = %d/%d\n', ...
    sum(significant_VSwitch_m3, 'omitnan'), numel(pVsVSwitch_VSwitchOnlyModel)-sum(isnan(pVsVSwitch_VSwitchOnlyModel)));


% Model Comparison Section
fprintf('\n=== Model Comparison Results ===\n');
% For each participant, find the index of the model with the minimum AIC/BIC
% 1=Full, 2=VRepeat only, 3=VSwitch only
[~, best_model_aic_idx] = min(AICs, [], 2);
[~, best_model_bic_idx] = min(BICs, [], 2);

% because 'min' returns index 1 for rows that are all NaN (i.e., skipped participants).
% We must manually set these back to NaN so they are not counted.
skipped_bic_rows = all(isnan(BICs), 2);
best_model_bic_idx(skipped_bic_rows) = NaN;

skipped_bic_rows = all(isnan(BICs), 2);
best_model_bic_idx(skipped_bic_rows) = NaN;

% Count how many times each model was the best, handling potential NaNs
% The isnan() check now correctly excludes the skipped participants
valid_indices_aic = ~isnan(best_model_aic_idx);
valid_indices_bic = ~isnan(best_model_bic_idx);

aic_counts = accumarray(best_model_aic_idx(valid_indices_aic), 1, [3 1]);
bic_counts = accumarray(best_model_bic_idx(valid_indices_bic), 1, [3 1]);

fprintf('Based on AIC, number of participants best fit by:\n');
fprintf('  Full Model (VRepeat + VSwitch): %d\n', aic_counts(1));
fprintf('  VRepeat Only Model:             %d\n', aic_counts(2));
fprintf('  VSwitch Only Model:             %d\n', aic_counts(3));

fprintf('\nBased on BIC, number of participants best fit by:\n');
fprintf('  Full Model (VRepeat + VSwitch): %d\n', bic_counts(1));
fprintf('  VRepeat Only Model:             %d\n', bic_counts(2));
fprintf('  VSwitch Only Model:             %d\n', bic_counts(3));

fprintf('\n(Note: Total counts may not sum to %d if some models failed to fit.)\n', ...
    nParticipants - numel(skipped_participants));

%% How many are COMPARATORS?

%  participants with significant NEGATIVE B_switch
is_sig_neg_vswitch = significant_VSwitch_full & (betasVSwitch < 0);
num_sig_neg_vswitch = sum(is_sig_neg_vswitch);

fprintf('Total participants with significant NEGATIVE B_switch (from Full Model): %d\n', num_sig_neg_vswitch);

% participants with significant POSITIVE B_repeat
is_sig_pos_vrepeat = significant_VRepeat_full & (betasVRepeat > 0);
num_sig_pos_vrepeat = sum(is_sig_pos_vrepeat);

fprintf('Total participants with significant POSITIVE B_repeat (from Full Model): %d\n', num_sig_pos_vrepeat);

% Who are the the OVERLAPPING participants (the "comparators")
% i.e.  These are participants who are TRUE for BOTH conditions
is_comparator = is_sig_neg_vswitch & is_sig_pos_vrepeat;
num_comparators = sum(is_comparator);
comparator_indices = find(is_comparator);

fprintf('Of the %d participants with a sig. negative B_switch,\n', num_sig_neg_vswitch);
fprintf('... %d ALSO have a sig. positive B_repeat.\n', num_comparators);
fprintf('This "comparator" group (n=%d) includes participant indices: %s\n', num_comparators, mat2str(comparator_indices));

%% How many are positive for VRepeat and non-significant for VSwitch??

% participants with significant POSITIVE B_repeat
is_sig_pos_vrepeat = significant_VRepeat_full & (betasVRepeat > 0);

% participants with NON-significant VSwitch
is_not_sig_vswitch = ~significant_VSwitch_full;

% OVERLAP of these two groups
is_pos_repeat_and_nonsig_switch = is_sig_pos_vrepeat & is_not_sig_vswitch;


num_pos_repeat_and_nonsig_switch = sum(is_pos_repeat_and_nonsig_switch);
fprintf('Participants with Sig Positive VRepeat AND Non-Sig VSwitch: %d\n', num_pos_repeat_and_nonsig_switch);

%% Plot pie charts for significant betas (VSwitch first)

customColors_VSwitch_4 = [
    0.25 0.55 0.85;  % Blue (Sig Pos)
    0.45 0.50 0.20;  % (comparators)
    0.95 0.60 0.50;  % Lighter Red (sig neg (non comparators))
    0.00 0.25 0.50   % dark blue (Non-Sig)
];

is_sig_pos_VRepeat = (betasVRepeat > 0 & significant_VRepeat_full);
is_sig_neg_VRepeat = (betasVRepeat < 0 & significant_VRepeat_full);
is_non_sig_VRepeat = ~significant_VRepeat_full;

is_sig_pos_VSwitch = (betasVSwitch > 0 & significant_VSwitch_full);
is_sig_neg_VSwitch = (betasVSwitch < 0 & significant_VSwitch_full);
is_non_sig_VSwitch = ~significant_VSwitch_full;

% Pie Chart for VSwitch
X_VSwitch = [
    sum(is_sig_pos_VSwitch), ... %blue (+beta)
    sum(is_sig_neg_VSwitch & is_sig_pos_VRepeat), ... %-Beta (comparators (dark red))
    sum(is_sig_neg_VSwitch & ~is_sig_pos_VRepeat), ... %-Beta (non-comparators) (light red))
    sum(is_non_sig_VSwitch) %non sig. 
];
total_VSwitch = sum(X_VSwitch);

if total_VSwitch > 0
    labels_VSwitch = { ...
        sprintf('Significant Positive (n=%d, %.1f%%)', X_VSwitch(1), (X_VSwitch(1)/total_VSwitch)*100), ...
        sprintf('Group 1: Sig. Neg. VSwitch & Sig. Pos. VRepeat (n=%d, %.1f%%)', X_VSwitch(2), (X_VSwitch(2)/total_VSwitch)*100), ...
        sprintf('Group 2: Sig. Neg. VSwitch & (Neg/Non VRepeat) (n=%d, %.1f%%)', X_VSwitch(3), (X_VSwitch(3)/total_VSwitch)*100), ...
        sprintf('Non-Significant (n=%d, %.1f%%)', X_VSwitch(4), (X_VSwitch(4)/total_VSwitch)*100) ...
    };
    legend_labels_VSwitch = {'Sign. Beta pos.', 'Comparators (Sig. Neg. VSwitch & Sig. Pos. VRepeat)', ...
        'Other Sig. Neg. VSwitch (non comparators)', 'Non-Significant'};
    
    figure;
    h_pie = pie(X_VSwitch, X_VSwitch > 0, labels_VSwitch);
    title('Distribution of All Betas (VSwitch)', 'FontSize', 14);
    colormap(customColors_VSwitch_4);
    set(findobj(h_pie, 'Type', 'text'), 'FontSize', 10, 'Color', 'black');
    legend(legend_labels_VSwitch, 'Location', 'eastoutside', 'FontSize', 10);
    set(gcf, 'Position', [100 100 800 600]);
end

%% VRepeat now

colors_VRepeat = [
    0.25 0.55 0.85;  % Blue (Sig Pos)
    0.45 0.50 0.20;  % (comparators)
    0.95 0.60 0.50;  % Lighter Red (sig neg (non comparators))
    0.00 0.25 0.50   %  (Non-Sig)
];

X_VRepeat_Revised = [
    sum(is_sig_pos_VRepeat & ~is_sig_neg_VSwitch), ...     % Sig Pos VRepeat (beta pos) (no comparators
    sum(is_sig_neg_VSwitch & is_sig_pos_VRepeat), ...      % comparators (Neg Repeat & Pos Switch)
    sum(is_sig_neg_VRepeat & ~is_sig_pos_VSwitch), ...     % sig neg. beta
    sum(is_non_sig_VRepeat)                                % 5. Non-Significant VRepeat
];

total_VRepeat_Revised = sum(X_VRepeat_Revised);

if total_VRepeat_Revised > 0
    labels_VRepeat_Revised = { ...
        sprintf('Significant Positive VRepeat (n=%d, %.1f%%)', X_VRepeat_Revised(1), (X_VRepeat_Revised(1)/total_VRepeat_Revised)*100), ...
        sprintf('comparators: Neg VRepeat & Pos VSwitch (n=%d, %.1f%%)', X_VRepeat_Revised(2), (X_VRepeat_Revised(2)/total_VRepeat_Revised)*100), ...
        sprintf('other neg sig: Neg VRepeat & Other (n=%d, %.1f%%)', X_VRepeat_Revised(3), (X_VRepeat_Revised(3)/total_VRepeat_Revised)*100), ...
        sprintf('Non-Significant (n=%d, %.1f%%)', X_VRepeat_Revised(4), (X_VRepeat_Revised(4)/total_VRepeat_Revised)*100) ...
    };

    figure;
    h_pie = pie(X_VRepeat_Revised, X_VRepeat_Revised > 0, labels_VRepeat_Revised);
    title('Distribution of VRepeat', 'FontSize', 14);
    colormap(colors_VRepeat);
    set(findobj(h_pie, 'Type', 'text'), 'FontSize', 10, 'Color', 'black');
    set(gcf, 'Position', [100 100 800 600]);

    legend_labels_VRepeat_4 = {
    'sign pos (without comparators)', ...
    'comparators (Pos VRep & Neg VSw)', ...
    'sign neg vrepeat', ...
    'Non-Significant'
};
legend(legend_labels_VRepeat_4, 'Location', 'eastoutside', 'FontSize', 10);
end

%% Plot beta coefficients for VRepeat (Subdivided by VSwitch, with cross-over)
[sorted_betas_VRepeat, sortIdx_VRepeat] = sort(betasVRepeat);

is_cross_posVRepeat_negVSwitch = (is_sig_pos_VRepeat & is_sig_neg_VSwitch);

% logical indices
sorted_non_sig_VRepeat = is_non_sig_VRepeat(sortIdx_VRepeat);
sorted_sig_pos_VRepeat = is_sig_pos_VRepeat(sortIdx_VRepeat);
sorted_sig_neg_VRepeat = is_sig_neg_VRepeat(sortIdx_VRepeat);
sorted_sig_neg_VSwitch = is_sig_neg_VSwitch(sortIdx_VRepeat);
sorted_sig_pos_VSwitch = is_sig_pos_VSwitch(sortIdx_VRepeat);
sorted_cross_group = is_cross_posVRepeat_negVSwitch(sortIdx_VRepeat);

% subgroups
sorted_group1_neg = (sorted_sig_neg_VRepeat & sorted_sig_pos_VSwitch);
sorted_group2_neg = (sorted_sig_neg_VRepeat & ~sorted_sig_pos_VSwitch);

sorted_sig_pos_VRepeat(sorted_cross_group) = false;


figure('Position',[461,535,481,245]);
hold on;
title('Beta Coefficients for VRepeat (Subdivided by VSwitch, with Cross-Over Override)');
xlabel('Participant (sorted)');
ylabel('Beta Value');


% Non-significant (dark blue)
plot(find(sorted_non_sig_VRepeat), sorted_betas_VRepeat(sorted_non_sig_VRepeat), ...
    'o', 'MarkerEdgeColor', colors_VRepeat(4,:), 'MarkerFaceColor', colors_VRepeat(4,:), 'MarkerSize', 4);

% Significant Positive (Blue)
plot(find(sorted_sig_pos_VRepeat), sorted_betas_VRepeat(sorted_sig_pos_VRepeat), ...
    'o', 'MarkerEdgeColor', colors_VRepeat(1,:), ...
    'MarkerFaceColor', colors_VRepeat(1,:), 'MarkerSize', 4);

% Sig. Pos. VRepeat & Sig. Neg. VSwitch (Yellow-ish)
plot(find(sorted_cross_group), sorted_betas_VRepeat(sorted_cross_group), ...
    'o', 'MarkerEdgeColor', colors_VRepeat(2,:), ... 
    'MarkerFaceColor', colors_VRepeat(2,:), 'MarkerSize', 5);

% Sig Neg VRepeat & (Neg/Non VSwitch) (Lighter Red - Color 3)
plot(find(sorted_group2_neg), sorted_betas_VRepeat(sorted_group2_neg), ...
    'o', 'MarkerEdgeColor', colors_VRepeat(3,:), ...
    'MarkerFaceColor', colors_VRepeat(3,:), 'MarkerSize', 4);

% legend and other stuff
yline(0, 'k--', 'LineWidth', 1.5);
idx_approx_crossings = find(diff(sign(sorted_betas_VRepeat)));
if ~isempty(idx_approx_crossings)
    first_crossing_x = idx_approx_crossings(1);
    xline(first_crossing_x,'k--', 'LineWidth', 1.5, 'DisplayName', 'First Zero Crossing');
end

legend({ ...
    'Non-Significant', ...
    'Significant Positive (without comparators)', ...
    'comparators: Sig. Pos. VRepeat & Sig. Neg. VSwitch', ...
    'Sig. Neg. VRepeat', ...
    'beta = 0'
    }, 'Location', 'best', 'FontSize', 8);

ylim([-35 35]);
xlim([0 254]);
filename = ['betas_vRepeat.pdf'] ;
% saveas(gcf,fullfile(figpath,filename))
%% Same, but for VSwitch
[sorted_betas_VSwitch, sortIdx_VSwitch] = sort(betasVSwitch);

% Non-Significant VSwitch
sorted_non_sig_VSwitch = is_non_sig_VSwitch(sortIdx_VSwitch);

% Significant Positive VSwitch
sorted_sig_pos_VSwitch = is_sig_pos_VSwitch(sortIdx_VSwitch);

% Sig Neg VSwitch & Sig Pos VRepeat
sorted_group1_neg = (is_sig_neg_VSwitch(sortIdx_VSwitch) & is_sig_pos_VRepeat(sortIdx_VSwitch));

% Sig Neg VSwitch & (Neg or Non-sig VRepeat)
sorted_group2_neg = (is_sig_neg_VSwitch(sortIdx_VSwitch) & ~is_sig_pos_VRepeat(sortIdx_VSwitch));

figure('Position',[461,535,481,245]);
hold on;
title('Beta Coefficients for VSwitch (Subdivided by VRepeat)');
xlabel('Participant (sorted)');
ylabel('Beta Value');

% Non-significant (dark blue)
plot(find(sorted_non_sig_VSwitch), sorted_betas_VSwitch(sorted_non_sig_VSwitch), ...
    'o', 'MarkerEdgeColor', customColors_VSwitch_4(4,:), 'MarkerFaceColor', customColors_VSwitch_4(4,:), 'MarkerSize', 4);

% Significant Positive (Blue)
plot(find(sorted_sig_pos_VSwitch), sorted_betas_VSwitch(sorted_sig_pos_VSwitch), ...
    'o', 'MarkerEdgeColor', customColors_VSwitch_4(1,:), ...
    'MarkerFaceColor', customColors_VSwitch_4(1,:), 'MarkerSize', 4);

% Sig Neg (yellowish - Color 2)
plot(find(sorted_group1_neg), sorted_betas_VSwitch(sorted_group1_neg), ...
    'o', 'MarkerEdgeColor', customColors_VSwitch_4(2,:), ...
    'MarkerFaceColor', customColors_VSwitch_4(2,:), 'MarkerSize', 4);

% Sig Neg (Lighter Red - Color 3)
plot(find(sorted_group2_neg), sorted_betas_VSwitch(sorted_group2_neg), ...
    'o', 'MarkerEdgeColor', customColors_VSwitch_4(3,:), ...
    'MarkerFaceColor', customColors_VSwitch_4(3,:), 'MarkerSize', 4);

yline(0, 'k--', 'LineWidth', 1.5);
idx_approx_crossings = find(diff(sign(sorted_betas_VSwitch)));
% If there are any crossings, plot a vertical line at the first one
if ~isempty(idx_approx_crossings)
    first_crossing_x = idx_approx_crossings(1); % Get only the first crossing's x-coordinate
    xline(first_crossing_x,'k--', 'LineWidth', 1.5, 'DisplayName', 'First Zero Crossing');
end

legend({ ...
    'Non-Significant', ...
    'Significant Positive', ...
    'comparators: Sig. Neg. VSwitch & Sig. Pos. VRepeat', ...
    'Sig. Neg. VSwitch (non comparators)', ...
    'beta = 0'
    }, 'Location', 'best', 'FontSize', 8); % Reduced font size slightly for fit

ylim([-35 35]);
xlim([0 254]);

filename = ['betas_vSwitch.pdf'] ;
% saveas(gcf,fullfile(figpath,filename))

%% AIC & BIC plots

%colum 1 = full; colum 2 = vrepeat; column 3 = vswitch
sumAIC = nansum(AICs);
sumBIC = nansum(BICs);
sumNegLogLikelihood = -nansum(LogLikelihoods);
model_names = {'Full Model', 'VRepeat Only', 'VSwitch Only'};

figure('Position', [-1521,964,216,327]);
plot(sumAIC,'.-k','MarkerSize',20)
set(gca,'FontSize',14,'XTick',1:length(sumAIC),'XTickLabel',model_names);
xlim([0.5, length(sumAIC)+0.5])
tmp = ylim; ylim(tmp.*[0.99 1.01]);
ylabel('AIC')
xlabel('model')

figure('Position', [-1521,964,216,327]);
plot(sumBIC,'.-k','MarkerSize',20)
set(gca,'FontSize',14,'XTick',1:length(sumBIC),'XTickLabel',model_names);
xlim([0.5, length(sumBIC)+0.5])
tmp = ylim; ylim(tmp.*[0.99 1.01]);
ylabel('BIC')
xlabel('model')

% BIC weights
minBIC = min(sumBIC); % minimum BIC value
deltaBIC = sumBIC - minBIC; % difference for each model
relLikelihood = exp(-0.5 * deltaBIC); %  relative likelihood (proportional to exp(-1/2 * delta))
bicWeights = relLikelihood / sum(relLikelihood); % Normalize to get weights that sum to 1

fprintf('BIC Weights:\n');
for i = 1:length(model_names)
    fprintf('%s: %.4f\n', model_names{i}, bicWeights(i));
end


filename = ['bic_glms.pdf'] ;
% saveas(gcf,fullfile(figpath,filename))


%% BIC count how many participants are better explained by the different glm models?
%uncomment stuff to get AIC instead

model_names = {'Full Model', 'VRepeat Only', 'VSwitch Only'};
% aic_counts = aic_counts';
% counts = [aic_counts]';
bic_counts = bic_counts';
counts = [bic_counts]';

figure('Position',[-1521,964,216,327]);
b = bar(counts);
ylabel('Number of Participants');
title('Model Comparison Results');
set(gca, 'xticklabel', model_names);
legend({'BIC'}, 'Location', 'northwest');
ylim([0 200]);

for i = 1:length(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(b(i).YData);
    text(xtips, ytips, labels, 'HorizontalAlignment','center',...
        'VerticalAlignment','bottom');
end
filename = ['nsubbic_glm.pdf'] ;
% saveas(gcf,fullfile(figpath,filename))

