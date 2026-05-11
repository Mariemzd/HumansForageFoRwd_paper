%% recover choice ll from params ;
clear;
cd('/Users/mac/Documents/MATLAB/ForagingByRichness/review/choiceLLAnalysis_2AB/')
addpath ../fitting/2AB_mturk/
addpath ../fitting/models/
addpath /Users/mac/Documents/MATLAB/general/
savepath = './2AB_mturk/' ;

load('singleiti_202203011023_lightweight.mat')
load('fitRLtoMTurk_20rounds_220724_0322.mat') %this is becket's fits
% load('fitRLto_singleiti_202203011023_lightweight.mat')
nsubs = length(trials);
ntrials = 300 ;
epsilon = 1e-100 ;

%%
LL=cell(nsubs,2);
averages = nan(nsubs,2);
probs=cell(nsubs,2);
for i=1:nsubs

    sideList = [trials(i).trials.choice]+1 ; %choice
    rewardList = [trials(i).trials.reward] ; %reward
    selex = find([trials(i).trials.practice]==0) ; %not practice
    rewardList = rewardList(selex) ;
    sideList = sideList(selex) ;
    
    switches = [NaN , diff(sideList) ~= 0] ;
    if nansum(switches)>1 

    theta = fits(1).params(i,:) ; % first foraging params 
    alpha = theta(1);
    rho = theta(2);
    beta = theta(3);
    v = 1; %initiate v_oit 

    probList = NaN(2,length(rewardList)); % 1st row is foraging, 2nd row standard rl. 
     
    
    for j=1:length(rewardList)
        
         
        if switches(j) == 1
            probList(1,j) =  1 / (1 + exp((v-rho)*beta)) ; 

            v=+ alpha*(rewardList(j)-v);

        elseif switches(j) == 0
            
             probList(1,j) = 1 - (1 / (1 + exp((v-rho)*beta))) ; 
            v= rho + alpha*(rewardList(j)-rho) ;
        end

        if j==1
        probList(1,j) = 0.5; 
        end 
    
    end
   

    theta = fits(2).params(i,:) ; %now select RL 

    alpha = theta(1);
    beta = theta(2);

    karm = max(sideList);
    Qvalue = zeros(karm,1)+1/karm; %initiate q-values
    

    for j=1:length(rewardList)

        arm = sideList(j) ;

        %Softmax function
% 
        % probList(2,j) = exp((Qvalue(arm)*beta)-LSE(Qvalue*beta)) ;
        
        probList(2,j) = exp(Qvalue(arm)*beta) / sum(exp(Qvalue.*beta));

        Qvalue(arm) =+ alpha*(rewardList(j)-Qvalue(arm)) ; 


    end

 
    % trials(i).choiceLogLike = log(probList+epsilon) ;
    trials(i).choiceLogLike = log(probList+epsilon) ;
    trials(i).probs = probList ;
    
    averages(i,:) = mean(probList,2) ; 
    if any(log(probList+epsilon) > 0) %sanity check 
        keyboard 
    end 
    end
    
    
    
end




%%
% filename = strcat('singleiti_202203011023_lighweight_choiceLL_RLvsForaging',datestr(datetime,'yymmdd'),'.mat');
filename = strcat('singleiti_202203011023_lighweight_choiceLL_RLvsForaging_og',datestr(datetime,'yymmdd'),'.mat');

save(filename, 'trials') 
