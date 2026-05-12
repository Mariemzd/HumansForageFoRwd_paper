function [theta,Zlabels,lik] = exp2mix(times,theta)
% fit a mixture of 2 exponentials
%  theta is fitted parameters:
%   theta(1) hazard of mix1
%   theta(2) hazard of mix2
%   theta(3) mixing proportion of mix1
%
% [theta,Zlabels,lik] = exp2mix(times,theta)

epsilon = 10^-6;
maxiter = 10^4;

t1 = mean(times(times <= (2/3)*max(times)+1));
t2 = mean(times(times >= (1/3)*max(times)+1));
t3 = 0.5; % start by assuming an equal mixture
tHat = [t1, t2, t3];

% setup our formula for the pdf
% now calculate prob that each obs belongs to class X, p 28 mT, last eq
fZ1 = @(x,tHat) (((tHat(1)/(1+tHat(1))).^x)*(tHat(3)/(1+tHat(1)))) ./ ...
    ((((tHat(1)/(1+tHat(1))).^x)*(tHat(3)/(1+tHat(1)))) + ...
    (((tHat(2)/(1+tHat(2))).^x)*((1-tHat(3))/(1+tHat(2)))));

if nargin < 2
    disp('starting EM');

    i = 0; stop = false;
    while ~stop

        % E-step
        Z1 = fZ1(times,tHat);
        Z2 = 1-Z1;

        % M-step
        t1 = sum(Z1.*times)/sum(Z1);
        t2 = sum(Z2.*times)/sum(Z2);
        t3 = mean(Z1);

        % update theta estimates for next round
        lastTHat = tHat;
        tHat = [t1 t2 t3];

        % but stop if we've reached our stopping criteria
        stop = sum(and(1-epsilon <= tHat./lastTHat,tHat./lastTHat <= 1+epsilon)) == length(tHat);
        stop = or(stop,i > maxiter);
        i = i+1;
    end
    
    theta = tHat;

    fprintf('\n EM finished after %d iterations',i)
    fprintf('\n mixture 1 hazard: %2.2f',theta(1))
    fprintf('\n mixture 2 hazard: %2.2f',theta(2))
    fprintf('\n mixture 1 weight: %2.2f \n',theta(3))
else
    fprintf('\n no EM conducted! just returning labels.\n')
end

% labels
Z1 = fZ1(times,theta);
Zlabels = Z1 >= 0.5; % keep the labels for putative short-bins

f2 = @(x) (theta(3))*(1./(1+theta(1)))*((theta(1)./(1+theta(1))).^x) + ...
    (1-theta(3))*(1./(1+theta(2)))*((theta(2)./(1+theta(2))).^x); % p 25 mT, middle eq
lik = sum(log(f2(times)));

% % save the fitted mixture function:
% f2 = @(x) (theta(3))*(1./(1+theta(1)))*((theta(1)./(1+theta(1))).^x) + ...
%     (1-theta(3))*(1./(1+theta(2)))*((theta(2)./(1+theta(2))).^x); % p 25 mT, middle eq
% 
% % fZ1, shift the plotting over, but eval starting at 0
% h = plot(x+1,(theta(3))*f1(x,theta(1)),'--','LineWidth',1.25);
% h(2) = plot(x+1,(1-theta(3))*f1(x,theta(2)),'--','LineWidth',1.25);
% h(3) = plot(x+1,f2(x),'-','LineWidth',3);
% set(h,'Color',[.42 .72 .95])