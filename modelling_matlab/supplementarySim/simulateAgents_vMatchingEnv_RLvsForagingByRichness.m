clear;
cd('/Users/mac/Documents/MATLAB/ForagingByRichness/review/simulations/')
% addpath ./models/
addpath /Users/mac/Documents/MATLAB/ForagingByRichness/review/fitting/2AB_mturk/
addpath /Users/mac/Documents/MATLAB/ForagingByRichness/review/choiceLLAnalysis_2AB/
addpath /Users/mac/Documents/MATLAB/general/

rest = load('fitRLtoMTurk_20rounds_220724_0322.mat'); %(becket's fits)
 %adding WSLS %MZ 24Nov20025
load('/Users/mac/Documents/MATLAB/ForagingByRichness/review/fitting/2AB_mturk/fitRLtoMTurk_WSLS_20rounds_251124.mat')

%% manually adding WSLS to previous fits
fits(1:2)=rest.fits(1:2) ; 
fits(3)=WSLS.fits(13) ; 
%%
delete(gcp('nocreate')) ; 
% % % create the parallel pool to use:
parpool(11); % leave me some CPUs, I have 12

%% pull in the fits to get the right parameter combinations:

% this will not do grid search, it will do matching parameters + matching
% environments

% we'll use the vanilla models, first grab env parameters
rlModelID = find(strcmp({fits.modelName},'standardRL'));
rlParams = [fits(rlModelID).params(:,1:fits(rlModelID).nParams)];
% params go alpha, beta

% now foraging parameters
fModelID = find(strcmp({fits.modelName},'foraging'));
fParams = [fits(fModelID).params(:,1:fits(fModelID).nParams)];
% params go alpha, thresh, beta

% now WSLS parameters %MZ oct2025
wModelID = find(strcmp({fits.modelName},'WSLS'));
wParams = [fits(wModelID).params(:,1:fits(wModelID).nParams)];

% be sure to remove NaN subjects
goodSubjects = and(sum(isnan(rlParams),2)==0,sum(isnan(fParams),2)==0);
rlParams = rlParams(goodSubjects,:);
fParams = fParams(goodSubjects,:);
wParams = wParams(goodSubjects,:);

% now we want to grab the values as well
load('singleiti_202203011023_lightweight.mat')

goodSubjects = find(goodSubjects);
V = cell(length(goodSubjects),1); %reward schedule
for k = 1:length(goodSubjects)
    V{k} = trials(k).header.reward_structure(:,[trials(k).trials.practice]==0);
end


%% now let's think about some simulations

env.nSessionsPerTheta = 100; % number of sessions to simulate at each theta value

rng(1); % seed random

% what are the parameters of our environment?
env.nTrials = length(V{1}); % this is the actual number, gives us a better match
env.nArms = 2;
env.hazard = 0.1; % p of step
env.stepSize = 0.1; % size of step
env.rwdBounds = [0.1,0.9];
rwdOpts = [env.rwdBounds(1):env.stepSize:env.rwdBounds(end)];

% alternatively, we can just grab all of our subjects directly and generate
% data that matches their actual parameters:
fParamOpts = fParams;
rlParamOpts = rlParams;
wParamOpts = wParams;

% now count the number of params we're going to use
nF = size(fParamOpts,1); % number of foraging models
nRL = size(rlParamOpts,1); % number of RL models
nWSLS = size(wParamOpts,1); % number of RL models

tic,

%%
% then iterate
parfor iter = 1:env.nSessionsPerTheta
% for iter = 1:env.nSessionsPerTheta
    allOut = [];
    % now parfor the foraging model
    for model = 1:3
    % try
        switch model
            case 1 % foraging
                [explores,choices,rewarded,times,params,type,transmat] = deal(cell(nF,1));
                for k = 1:nF
                    mdl = struct('rewards',V{k},'alpha',fParamOpts(k,1),'thresh',fParamOpts(k,2),'beta',fParamOpts(k,3),...
                        'nArms',env.nArms,'nTrials',env.nTrials);
                    params{k} = [mdl.alpha, mdl.beta, mdl.thresh]; % CAUTION! we're changing the order here
                    type{k} = 'f';

                    % generate all the behavior
                    [choices{k},rewarded{k}] = simRLforaging(mdl);

                    if length(unique(choices{k})) > 1 % if they chose more than 1 option
                       
                        % save the times
                        times{k} = diff(find(diff(choices{k})~=0));
                    else
                        explores{k} = zeros(size(choices{k}));
                        times{k} = [];
                    end
                end
       
            case 2    
                % now parfor the RL model
                [explores,choices,rewarded,times,params,type] = deal(cell(nRL,1));
                for k = 1:nRL
                    mdl = struct('rewards',V{k},'alpha',rlParamOpts(k,1),'beta',rlParamOpts(k,2),...
                        'nArms',env.nArms,'nTrials',env.nTrials);
                    params{k} = [mdl.alpha, mdl.beta];
                    type{k} = 'rl';

                    % generate all the behavior
                    [choices{k},rewarded{k}] = simRLvanilla(mdl);

                    if length(unique(choices{k})) > 1 % if they chose more than 1 option
                
                         
                        % save the times
                        times{k} = diff(find(diff(choices{k})~=0));
                    else
                        explores{k} = zeros(size(choices{k}));
                        times{k} = [];
                    end
                end
        
            case 3 
                [explores,choices,rewarded,times,params,type,transmat] = deal(cell(nWSLS,1));
                for k = 1:nWSLS
                    mdl = struct('rewards',V{k},'epsilon',wParamOpts(k,1),...
                        'nArms',env.nArms,'nTrials',env.nTrials);
                    params{k} = [mdl.epsilon];
                    type{k} = 'WSLS';

                    % generate all the behavior
                    [choices{k},rewarded{k}] = simWSLS(mdl);

                    if length(unique(choices{k})) > 1 % if they chose more than 1 option
                        
                        % save the times
                        times{k} = diff(find(diff(choices{k})~=0));
                    else
                        explores{k} = zeros(size(choices{k}));
                        times{k} = [];
                    end
                end 
        
        end

        % calculate some summary stats
        p_rwd = arrayfun(@(k) mean(rewarded{k}),1:length(choices),'UniformOutput',false)';
        p_rwd_chance = arrayfun(@(k) mean(rewarded{k})-mean(mean(V{k})),...
            1:length(choices),'UniformOutput',false)';
        p_best = arrayfun(@(k) mean(V{k}(sub2ind(size(V{k}),choices{k},1:env.nTrials))==max(V{k})),...
            1:length(choices),'UniformOutput',false)';
        p_switch = arrayfun(@(k) mean(diff(choices{k})~=0),1:length(choices),'UniformOutput',false)';
        p_explore = arrayfun(@(k) mean(explores{k}),1:length(choices),'UniformOutput',false)';
        
        out = struct('type',type,'params',params,'choices',choices,'objV',V,...
            'rewards',rewarded,'explores',explores,'times',times,...
            'p_rwd',p_rwd,'p_rwd_chance',p_rwd_chance,'p_best',p_best,...
            'p_switch',p_switch,'p_explore',p_explore,'transmat',transmat);

     
        allOut = [allOut; out]; 

    end
    
    % then save for this iteration
    if iter < 10
        zPad = '00';
    elseif iter < 100
        zPad = '0';
    else
        zPad = '';
    end
    out = allOut;
    fNumStr = strcat(zPad,num2str(iter));
    parsave(strcat('simResults',fNumStr),out)
    
    % this means that each 'out' will be a complete set of simulations for
    % both RL and foraging agents in 1 specific environment, we can then
    % concatenate them to put together many simulations all at once

end
toc

