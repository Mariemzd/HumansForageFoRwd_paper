function result = bootstrap_diffvect(vect,nPerm)

% tests whether the observed mean of vect of differences
% is significantly different from a null model.
% vect = vector to bootstrap 
% nPerm % number of permutations for null model
% Requires: generate_random_vector(n, nPerm)

% MZ 2025 
% -------------------------------------------------------------------------

% --- Validate input ---
if nargin < 1 || isempty(vect) || all(isnan(vect))
    result = struct('choseBest', NaN, ...
        'choseWorst', NaN, ...
        'observedMean', NaN, ...
        'CI', [NaN, NaN]);
    return

end

if nargin <2
    nPerm = 10000;  % number of permutations for null model
end

  
diffa = vect(~isnan(vect));

null_model = generate_random_vector(length(diffa), nPerm);

test = diffa .* null_model;
null_mean = mean(test, 1);
observed_mean = mean(diffa, 'omitnan');

% confidence interval
CI = prctile(null_mean, [2.5, 97.5]);

%output 
result = struct( ...
    'choseBest',  observed_mean > CI(2), ...
    'choseWorst', observed_mean < CI(1), ...
    'observedMean', observed_mean, ...
    'CI', CI);

end
