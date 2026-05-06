function [correction] = HolmBonferroni(p,sigCrit)
%   accepts a vector of p values (p) and significance threshold (sigCrit),
%   applies the Holm-Bonferroni correction to the significance criteria
%   returns vector of sig. criteria for each of the p-values in the vector,
%   corrected for multiple comparisons
%
%   example:
%
%   ps = [.05,.01,.01,.02];
%   correctedPs = HolmBonferroni(ps,0.05);
%
%   for hypothesis testing:
%       ps < HolmBonferroni(ps,0.05)
%
    toShape = 0;
    
    % if we got a column vector instead of row
    if size(p,1) == length(p)
        toShape = 1;
        p = p';
    end

    % only correct for sucessfully performed tests
    holdNaNs = find(~isnan(p));
    fu = p(~isnan(p));
    
    % sort and make correction vector
    [sorted,idx] = sort(fu);
    correction = fliplr(1:length(sorted));
    
    % gives you p values in order
    sorted = sigCrit ./ correction;

    % then transform back
    sorted(idx) = sorted;
    
    correction = NaN(1,length(p));
    correction(holdNaNs) = sorted;
    
    % flip back
    if toShape
        correction = correction';
    end

end