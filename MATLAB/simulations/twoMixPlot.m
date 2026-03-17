function [fH,th2] = twoMixPlot(times,xMax)

if nargin < 2; xMax = 40; end

x = [0:0.5:xMax];
xbins = [0:xMax];

% save the fitted mixture function:
f2 = @(x,theta) (theta(3))*(1./(1+theta(1)))*((theta(1)./(1+theta(1))).^x) + ...
    (1-theta(3))*(1./(1+theta(2)))*((theta(2)./(1+theta(2))).^x); % p 25 mT, middle eq

fH = figure(); clear h; hold on;
set(gca,'FontSize',16);

theta = exp2mix(times);
y = histc(times,xbins)./length(times);

% y = histc(times,x); % trying for plot in pdf space
% y = y./trapz(x,y); % nope - not for the discrete distribution

bar(xbins+.5,y,'FaceColor',[.5 .5 .5],'LineStyle','none',...
    'BarWidth',1);

% fZ1, shift the plotting over, but eval starting at 0
h = plot(x+.5,f2(x,theta),'-','LineWidth',3); hold on; 
set(h,'Color',[.42 .72 .95])

th2 = theta

% now the maximum likelihood version of the same
theta = nanmean(times); % mle estimate, eq 2.3 mT
% marginal distribution of times: (eq 2.1 mT)
f1 = @(x,theta) (1./(1+theta))*((theta./(1+theta)).^x);
h(2) = plot(x+.5,f1(x,theta),'--k','LineWidth',3);

th = plot(x+.5,th2(3)*f1(x,th2(1)),'--');
th(2) = plot(x+.5,(1-th2(3))*f1(x,th2(2)),'--');
set(th,'Color',[.42 .72 .95])

xlim([min(x) max(x)])

legend(h,'2mix','1exp')