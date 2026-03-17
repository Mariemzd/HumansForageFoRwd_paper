clear; close all ;
cd /Users/mac/Documents/MATLAB/ForagingByRichness/review
addpath fitting/
load volatilityLMH_newstates2025.mat
figpath = './figs/';
%% Supplementary figure S6
% will generate supplementary figure S6D unless : 
onset = 0; %1 for supplementary figure S6.B
ore = 0; % 1 for supplementary figure S6.A
oit = 0; 

m = NaN(length(trials),5); 
n = NaN(length(trials),5);
p = NaN(length(trials),5);

for i=1:length(trials)

    if isempty(trials(i).state2025)
        %skip people that where HMM didn't fit
        continue
    end

    selex = [trials(i).trials.practice]'==0 ;
    good = trials(i).trials(selex); %don't keep practice
    choice = [good.choice]+1;


    if (length(unique(choice))==4 )%because there are bug choices this a way to get rid of them
        z = z+1;
        choice(choice==4) = nan ;
    end

    switches = diff(choice)~=0 ;

    explore = [trials(i).state2025]==1;
    exploit = [trials(i).state2025]~=1 ;
    switches = [false ,switches]; %did they switch option in the current trial
    onsets = [true, diff([trials(i).state2025]==1)==1] ;
    offsets_oit = [diff(exploit)==-1,false]; 
   
  
        
    obj_rwd=NaN(length(choice),max(choice)); %extract reward seed
    other_rwd=NaN(length(choice),1);
    curr_rwd=NaN(length(choice),1);

    last_exploited_choice = 0 ; 
    
    switches_to_last_exploited_option=NaN(1,length(choice)); 

    for r=2:length(choice)
        if offsets_oit(r)==1
            last_exploited_choice = choice(r); 
        end 

        other_idx = setdiff([1:max(choice)],[choice(r),choice(r-1)]) ;

        if isscalar(other_idx) && choice(r)<=length(good(r).reward_seed) %only when they switch ; second condition is a sanity check
            other_rwd(r) = good(r).reward_seed(other_idx); %store obj value of alternative option (they didn't choose)
            curr_rwd(r) = good(r).p_reward ;%store obj value of option they chose
             
         
            switches_to_last_exploited_option(r) = choice(r)==last_exploited_choice; 

           
        end

    end
    
    trials(i).switch_last_oit = switches_to_last_exploited_option;
    difference = curr_rwd-other_rwd;
    switches_to_last_exploited_option(isnan(switches_to_last_exploited_option)) = 0;


   
    trials(i).switch_diff = difference; 
    trials(i).switches= nansum(switches) ;

    if onset
        cond = switches&explore&onsets&~switches_to_last_exploited_option ; 
    elseif ore
         cond = switches&explore&~switches_to_last_exploited_option;
    elseif oit 
         cond = switches&exploit&~switches_to_last_exploited_option;
    else
       cond = switches;
    end

    trials(i).switch_diff(~cond) = NaN ; 
    
    nboot = 50000; 
    
    res = bootstrap_diffvect(trials(i).switch_diff(~switches_to_last_exploited_option),nboot) ; 
    m(i,:) = [res.observedMean, res.choseBest, res.choseWorst, res.CI];
        
  
    resoit = bootstrap_diffvect(difference(switches&exploit),nboot) ; 
    n(i,:) = [resoit.observedMean, resoit.choseBest, resoit.choseWorst, resoit.CI];

    resore = bootstrap_diffvect(difference(switches&explore),nboot) ; 
    p(i,:) = [resore.observedMean, resore.choseBest, resore.choseWorst, resore.CI];



    trials(i).zebi = sum(switches_to_last_exploited_option(switches & exploit)) / sum(switches & exploit); 
    
end



%%
figure('Position',[476,470,476,310]);
norm = ['count'];
edges = [-0.4:0.8/20:0.4] ;

y1 = m(:,1) ;  %observed means
y2 = y1(m(:,2)==1) ;%chose best 
y3 = y1(m(:,3)==1) ; %chose worst 

h1 = histogram(y1,[edges],'FaceColor','#ebedf3',LineWidth=1.5,Normalization=norm) ; hold on ;
h2 = histogram(y2,[edges],'FaceColor','#00537a',LineWidth=1.5,Normalization=norm);
h3 = histogram(y3,[edges],'FaceColor','#ffaa00',LineWidth=1.5,Normalization=norm);
xline(0,'--',LineWidth=1.5)
ylim([0,80])
formatAxes(gca,0)
ylabel(norm)
xlabel('current - alternative')
legend([h1,h2,h3],{'all participants', sprintf('chose best = %d', nansum([m(:,2)])), ...
    sprintf('chose worst = %d', nansum([m(:,3)]))}, 'Location','NorthWest')
if onset
    name = 'explore onset' ;
elseif ore
    name ='explore';
elseif oit
    name ='exploit';
else
    name='switch';
end
title(name)

filename = ['histogram_difff',name,'without_last_exploited','50kboots_RLvsForaging.pdf'] ;
saveas(gcf,fullfile(figpath,filename))


