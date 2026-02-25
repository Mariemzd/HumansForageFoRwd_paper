% choiceLL, 1st is foraging 2nd is standard RL
% All analyses found are meant to explore whether people change strategy
% during the session : from compare alternatives to compare to threshold
% for example. % supplementary figure 3

clear; close all ;
cd('/Users/mac/Documents/MATLAB/ForagingByRichness/review/choiceLLAnalysis_2AB/')
addpath ../fitting/2AB_mturk/
addpath ../fitting/models/
addpath /Users/mac/Documents/MATLAB/general/
savepath = './2AB_mturk/' ;

load('singleiti_202203011023_lighweight_choiceLL_RLvsForaging_og251010.mat')

load fitRLtoMTurk_20rounds_220724_0322.mat %Becket's fits 


figpath = '/Users/mac/Documents/MATLAB/ForagingByRichness/review/figs/';

nsubs = length(trials);
m=NaN(nsubs,2,2);
fifty=NaN(nsubs,2,2);
n= NaN(nsubs,2,599);  
cll = NaN(nsubs,2,300) ;
probs = NaN(nsubs,2,300) ;
format = 'pdf';

mStr = {'foraging', 'RL'} ; 
modCol = {[0.0, 0.6, 0.2],[0.6, 0.5, 0.1]};
mStr = {'foraging', 'RL'} ; 

modedgecol = {[0.0, 0.6, 0.2],[0.6, 0.5, 0.1] };

%% extract choiceLL
for s=1:nsubs
    
    choiceLL = trials(s).choiceLogLike ; %foraging 1sT
    choiceprob = trials(s).probs ; 
    half = length(choiceLL)/2;
    try 
    m(s,:,1) = nansum(choiceLL(:,1:half),2) ; %1st half
    m(s,:,2) = nansum(choiceLL(:,half:end),2) ; %2nd half 
    cll(s,:,:) = choiceLL;
    probs(s,:,:) = choiceprob ; 
    fifty(s,:,1) = nansum(choiceLL(:,1:25),2) ; %1st 50
    fifty(s,:,2) = nansum(choiceLL(:,25:end),2) ; %2nd 50



    catch 
        s %basically the 4 participants that didn't switch
    end 

 
end 


%% unity plot AIC (FOR - RL) in 1st 50 vs last 50 trials
ms=150;
lw = 1.5 ;
figure ; 
firstaic(:,1)= aicbic(fifty(:,1,1),[fits(1).nParams]) ; %50 first trials foraging 
firstaic(:,2)= aicbic(fifty(:,2,1),[fits(2).nParams]) ; %50 first trials RL 
lastaic(:,1)= aicbic(fifty(:,1,2),[fits(1).nParams]) ; %50 last trials foraging 
lastaic(:,2)= aicbic(fifty(:,2,2),[fits(2).nParams]) ; %50 last trials RL
yline(0,LineWidth=lw) ; xline(0,LineWidth=lw) ; hold on ;


%%%%%9
scatter((firstaic(:,1)-firstaic(:,2)), (lastaic(:,1)-lastaic(:,2)),ms,'MarkerFaceColor','#0072BD','MarkerEdgeColor','none','MarkerFaceAlpha',0.5) ; 
formatAxes(gca,1)
xlabel(' AIC_{comp-thresh} - AIC_{comp-alt} \newline in first 50 trials')
ylabel(' AIC_{comp-thresh} - AIC_{comp-alt} \newline in last 50 trials')
% xlim([-1,1]*10^4)
filename = 'strategyChange_compareChoiceLL_RLvsForaging' ; 
saveas(gcf,fullfile(figpath,filename),format)
%manually compute aic 

% model improvement: (interpreted as % better
% http://ejwagenmakers.com/2004/aic.pdf
minaicf =  min(vertcat(firstaic),[],2) ; 
aicWf = exp(( minaicf-firstaic)/2) ; 

minaicl =  min(vertcat(lastaic),[],2) ; 
aicWl = exp((minaicl-lastaic)/2) ; 

bestff = nansum(aicWf(:,1)==1); %chose foraging first 

bestlll = nansum(aicWl(:,1)==1);%chose foraging last

switchedfr = nansum(aicWf(:,1)==1 & aicWl(:,2)==1) %started foraging then moved to rl 
switchedrl = nansum(aicWf(:,2)==1 & aicWl(:,1)==1) %started rl then moved to foraging
switchers_idx = [find(aicWf(:,1)==1 & aicWl(:,2)==1);find(aicWf(:,2)==1 & aicWl(:,1)==1)]




%% Now plot LL + smoothe it
close; 
bnn = 5; 
clear y 
figure('Position',[476,360,560,420]) ; hold on
 
y(1,:)=squeeze(nanmean(probs(:,1,:),1))' ; e(1,:)= squeeze(nanste(probs(:,1,:),1))';%foraging 
y(2,:)=squeeze(nanmean(probs(:,2,:),1))' ; e(2,:)= squeeze(nanste(probs(:,2,:),1))';%RL 

edges = 1:10:300; 

% keyboard
for i=1:2

    x = 1:length(y(i,:)) ; 
  
     [xfit, yfit] = kreg(x,y(i,:),bnn,kernel='gauss'); 
     efit = kreg(x,e(i,:),bnn);

    shadedline(xfit, yfit, efit,"Color",modCol{i},...
    "LineWidth",2)

 
    smoothedY(i,:) = yfit ; 
end

set(gca,'FontSize',20,...
    'Fontname','arial',...
    'XColor','k','YColor','k','ZColor','k','Layer','top','LineWidth',1.5);
box off

ylabel('p (choice | \theta_{m},m)')
xlabel('trials')
legend({'comp-thresh','','comp-alt'},"Location","southeast")
% ylim([0.45,.85])
ax = gca;
ax.Toolbar.Visible = 'off';


filename= 'plotcprob_compareChoiceLL_RLvsForaging' ;
saveas(gcf,fullfile(figpath,filename),format)

