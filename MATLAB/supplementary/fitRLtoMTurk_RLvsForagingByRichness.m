clear;
cd('/Users/mac/Documents/MATLAB/ForagingByRichness/review/fitting/')
addpath ./2AB_mturk/
addpath ./models/
addpath /Users/mac/Documents/MATLAB/general/
savepath = './2AB_mturk/' ;

fname = 'singleiti_202203011023_lightweight' ;
load(fname)

%%  

nIter = 20; % number of iterations to fit for
maxBeta = 100; % highest value that beta can take on
maxThresh = 2 ; 
minThresh = 0 ; 
rng(1); % seed random

% "preallocate"
fits = [];
tic
for model = 2:4  
    switch model
        case 1 % foraging
            nParams = 4; % how many parameters is the function expecting?
            nanParams = [0,0,0,1]; % logical, which to avoid fitting
            fits(model).modelName = 'foraging';
            betaSlot = 3;
            threshSlot= 2;
        case 2 % vanilla RL
            nParams = 4;
            nanParams = [0,0,1,1];
            fits(model).modelName = 'standardRL';
            betaSlot = 2;

        case 3 %WSLS
            nParams = 1;
            nanParams = [0,1,1,1];
            fits(model).modelName = 'WSLS';
            betaSlot = NaN;

        case 4 %repetition bonus RL
            nParams = 3;
            nanParams = [0,0,0,];
            fits(model).modelName = 'rbRL';
           betaSlot = 3;
       
    end

    % spit the model name to the command line
    fits(model).modelName

    % initialize and pre-allocate
    fits(model).params = NaN(length(trials),nParams+5);
    fits(model).lik = 0;
    fits(model).nChoices = 0;
    fits(model).nParams = sum(nanParams==0);

    for k = 1:length(trials)

        % pull out the data
        tmp = trials(k).trials;
        selex = [trials(k).trials.practice]==0; % exclude practice trials
        t.choice = [tmp(selex).choice]+1;
        t.reward = [tmp(selex).reward];
        % t.explore = [tmp(selex).states]==1;
    
        % pre-allocate model's parameters to be fitted
        mlpar = NaN(1,nParams+5);

        if length(unique([t.choice])) == 2 % if we get some choices

            % figure out what model to use
            switch model
                case 1
                    f = @(params) model_Foraging(t,params,maxBeta);
                case 2
                    f = @(params) model_RLvanilla(t,params,maxBeta);
                case 3
                    f = @(params) model_WSLS(t,params,maxBeta);
                case 4
                    f = @(params) model_RLrepetitionBonus(t,params,maxBeta);


            end

            if k == 1
                f
            end
        
            % fitting options
            options = optimset('MaxFunEvals',2000,'MaxIter',1000,'Display','none',...
                'TolX',10^-4,'TolFun',10^-4,'OutputFcn',@manualOptimStop); %MZ oct2025 added manual output fx
            
            % initialize minima
            fvalmin = Inf;
     

            for i = 1:nIter
                % learning  noise
                initP = rand(1,nParams); % make a guess for this iteration
                initP(nanParams==1) = NaN; % don't fit the things we don't want to fit
                %Exponential draw for beta %MZ added to match rest of
                %analysis oct 2025

                if nParams>1
                initP(betaSlot) =  random('Exponential', 1) ;  %initP(betaSlot).*maxBeta; % get beta closer to the right scale
                end 

                if model == 1
                    initP(threshSlot) = randb(1,0,maxThresh); %bounded initial sample
                end
        
                [guess , fval , exitflag , output] = fminsearch(f,initP,options);
                
                if fval <= fvalmin && ~isinf(fval) % save output if we get improvement %%&& exitflag == 1 %this is commented out after adding our own stopfx
                    fvalmin = fval; % replace previous minima
                    [lik,agreement] = f(guess); % re-run the model
                    mlpar(1:nParams) = guess(1:nParams);
                    mlpar(nParams+2) = lik; % likelihood
                    mlpar(nParams+3) = agreement; % model agreement
                    mlpar(nParams+4) = output.iterations;
                    mlpar(nParams+5) = exitflag; %problem with this in matlab version > 2022 so we added fx stopper
                end
            
            end
             
            fits(model).params(k,:) = mlpar;
            fits(model).lik = fits(model).lik + mlpar(nParams+2);
            fits(model).nChoices = fits(model).nChoices + length(t.choice);
        else
            fprintf('too few choices, skipping participant %2.0f\n',k)
            fits(model).params(k,:) = mlpar;
        end
    end

    fits(model).avgParams = nanmean(fits(model).params);
    fits(model).likelihood = fits(model).params(:,nParams+2);
    fits(model).agreement = fits(model).params(:,nParams+3);
end

fname = strcat('fitRLtoMTurk_20rounds_',datestr(datetime,'yymmdd'));
save(fname,'fits')
toc
