function [LLout,transmatOut,labelsOut,priorOut] = fitOreOitHMM(data,verbose)

if nargin < 2
    verbose = true;
end

%%
% initialize
nStates = 3;
nSeeds = 20; % number of times to reseed

% transition matrix:
transmat_logical = eye(nStates); % all-to-self OK
transmat_logical(1,:) = 1;
transmat_logical(:,1) = 1; % all to and from ORE
      
% emissions weights: enforced/fixed
emat = [.5, .5;...
         0,  1;...
         1,  0];

% initial state distribution:
initmat_logical = zeros(nStates,1); % start in ORE
initmat_logical(1) = 1;
 % only used if initSeeds are not provided

% tolerance and exiting stuff
MaxIter = 10^4; MaxdLL = 10^-4;

% process the sequence into the format we want it in
[observations,stateLabels] = deal(cell(size(data,1),1));
        % also preallocate the state labels
for seq = 1:size(data,1)

    obs = data{seq};
    choice = obs(1,:);
    reward = obs(2,:);
    
    observations{seq} = [choice' reward'];
end

fvalMin = Inf;
%%

figure(); hold on;
colors = {'r','b','m','g','k','c'};

for seed = 1:nSeeds
    
    % randomly seed the starting parameters
    transmat = mk_stochastic(transmat_logical.*rand(nStates,nStates));
%         transmat = [.9 .05 .05; .1 .9 0; .1 0 .9];
%           transmat = [.80 .1 .1; .25 .75 0; .25 0 .75]
    initmat = mk_stochastic(initmat_logical.*rand(nStates,1));



    % tie the parameters right off the bat
    transmat = tieParams(transmat);
    
    % initalize for EM
    LL = NaN; deltaLL = NaN; oldLL = -inf; iter = 0;

    while (1)

        iter = iter + 1; LL = 0;

        % E step
        % set up sufficient statistics for the HMM
        exp_num_trans = zeros(nStates,nStates);%+10^-10;
        exp_num_visits1 = zeros(nStates,1);%+10^-10;
        exp_num_visitsT = zeros(nStates,1);%+10^-10;

        % and containers for all our weights and such
        allChoices = []; allOutcomes = [];

        % step through ever sequence we observed
        for seq = 1:size(observations,1)

            Time = size(data{1},2);

            % pull in the data for this round
            obs = observations{seq};
            choices = obs(:,1)';
            rewards = obs(:,2)';
            Time = length(obs);

            % calculate the probability of each obs, given each state
         try
                p_obs = emat*dummyvar(choices)';
            catch
                tmp = false(2,size(choices,2));
                tmp(sub2ind(size(tmp),choices,1:size(choices,2))) = true;
                p_obs = emat*tmp;
            end

            if sum(sum(isnan(p_obs))) > 1;
                p_obs(isnan(p_obs)) = 0;
            end

            % now the joint probability of states and obs, given model
            [gamma,xi,currlik] = forwards_backwards(initmat, transmat, p_obs);
            % gamma is the posterior probabilitiy of the states
            %   p(yi | xt)
            % xi replicates the transition matrix

            % update the lik
            LL = LL+currlik;
            
            % now update the sufficient statistics

            % easy for the discrete nodes
            exp_num_trans = exp_num_trans + sum(xi,3);
            exp_num_visits1 = exp_num_visits1 + gamma(:,1);
            exp_num_visitsT = exp_num_visitsT + gamma(:,Time); % this is ci in the case of 1 mix

        end

        % update the likelihood
        deltaLL = oldLL-LL; % change in log likelihood
        oldLL = LL; % calculate log likelihood
        
        % give the console an update, if requested
        if verbose, fprintf(1, 'iteration %d, loglik = %f\n', iter, LL); end
        % stop now if we're not improving or we've iterated too much
        if (abs(deltaLL) < MaxdLL || iter > MaxIter); break; end

        % M step

        % first, we'll accomplish our parameter tying:
        exp_num_trans = tieParams(exp_num_trans);
        
%         % first, the self-excitations are tied:
%         exp_num_trans(2:end,2:end) = eye(nStates-1) .* ((sum(diag(exp_num_trans))-exp_num_trans(1,1))/2);
%         
%         % then the transitions into ORE
%         exp_num_trans(2:end,1) = deal(nanmean(exp_num_trans(2:end,1)));
%         
%         % then the transitions from ORE
%         exp_num_trans(1,2:end) = deal(nanmean(exp_num_trans(1,2:end)));

        % square up our discrete nodes:
        startprob = normalise(exp_num_visits1);
        endprob = normalise(exp_num_visitsT);
        transmat = mk_stochastic(exp_num_trans);

        initmat = startprob;
        
        if round(iter/10) == (iter/10) || iter == 1
            initmat
            transmat
        end
        
        %plot(iter,LL,'.','MarkerSize',10,'Color',colors{seed});
         plot(iter,LL,'.','MarkerSize',10,'Color',[0, 0.4470, 0.7410]);
        drawnow;
    end

    if iter>MaxIter
        warning('fitVonMixHMM:MaxIter','This iteration did not converge.');
    elseif LL < fvalMin
        disp('model improvement!')
        fvalMin = LL; % set the new minimum
        transmatOut = transmat; LLout = LL;
        labelsOut = stateLabels; priorOut = initmat;
    end
end

if isinf(fvalMin)
    warning('fitVonMixHMM:MaxIter','Baum-Welch did not converge!');
    transmatOut = NaN(nStates,nStates); LLout = NaN;
    labelsOut = NaN(size(stateLabels)); priorOut = initmat;
else
    disp('Baum-Welch converged just fine. Proceed with confidence!')
end


%% helpers

function mxOut = tieParams(mxIn)
% do the parameter tying where ever it's required:

    mxOut = mxIn;
    
    % first, the self-excitations are tied:
    mxOut(2:end,2:end) = eye(nStates-1) .* ((sum(diag(mxIn))-mxIn(1,1))/2);

    % then the transitions into ORE
    mxOut(2:end,1) = deal(nanmean(mxIn(2:end,1)));

    % then the transitions from ORE
    mxOut(1,2:end) = deal(nanmean(mxIn(1,2:end)));
 
end

function T = mk_stochastic(T)
    % MK_STOCHASTIC Ensure the argument is a stochastic matrix, i.e., the sum over the last dimension is 1.
    % T = mk_stochastic(T)
    %
    % If T is a vector, it will sum to 1.
    % If T is a matrix, each row will sum to 1.
    % If T is a 3D array, then sum_k T(i,j,k) = 1 for all i,j.

    % Set zeros to 1 before dividing
    % This is valid since S(j) = 0 iff T(i,j) = 0 for all j

    if (ndims(T)==2) & (size(T,1)==1 | size(T,2)==1) % isvector
      T = normalise(T);
    elseif ndims(T)==2 % matrix
      S = sum(T,2); 
      S = S + (S==0);
      norm = repmat(S, 1, size(T,2));
      T = T ./ norm;
    else % multi-dimensional array
      ns = size(T);
      T = reshape(T, prod(ns(1:end-1)), ns(end));
      S = sum(T,2);
      S = S + (S==0);
      norm = repmat(S, 1, ns(end));
      T = T ./ norm;
      T = reshape(T, ns);
    end
end

function [M, z] = normalise(A, dim)
    % NORMALISE Make the entries of a (multidimensional) array sum to 1
    % [M, c] = normalise(A)
    % c is the normalizing constant
    %
    % [M, c] = normalise(A, dim)
    % If dim is specified, we normalise the specified dimension only,
    % otherwise we normalise the whole array.

    if nargin < 2
      z = sum(A(:));
      % Set any zeros to one before dividing
      % This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
      s = z + (z==0);
      M = A / s;
    elseif dim==1 % normalize each column
      z = sum(A);
      s = z + (z==0);
      %M = A ./ (d'*ones(1,size(A,1)))';
      M = A ./ repmatC(s, size(A,1), 1);
    else
      % Keith Battocchi - v. slow because of repmat
      z=sum(A,dim);
      s = z + (z==0);
      L=size(A,dim);
      d=length(size(A));
      v=ones(d,1);
      v(dim)=L;
      %c=repmat(s,v);
      c=repmat(s,v');
      M=A./c;
    end
end

function p = approxeq(a, b, tol, rel)
    % APPROXEQ Are a and b approximately equal (to within a specified tolerance)?
    % p = approxeq(a, b, thresh)
    % 'tol' defaults to 1e-3.
    % p(i) = 1 iff abs(a(i) - b(i)) < thresh
    %
    % p = approxeq(a, b, thresh, 1)
    % p(i) = 1 iff abs(a(i)-b(i))/abs(a(i)) < thresh

    if nargin < 3, tol = 1e-2; end
    if nargin < 4, rel = 0; end

    a = a(:);
    b = b(:);
    d = abs(a-b);
    if rel
      p = ~any( (d ./ (abs(a)+eps)) > tol);
    else
      p = ~any(d > tol);
    end
end

function [gamma, xi, loglik] = forwards_backwards(prior, transmat, obslik, maximize)
    % FORWARDS_BACKWARDS Compute the posterior probs. in an HMM using the forwards backwards algo.
    % [gamma, xi, loglik] = forwards_backwards(prior, transmat, obslik, maximize)
    % Use obslik = mk_dhmm_obs_lik(data, b) or obslik = mk_ghmm_obs_lik(data, mu, sigma) first.
    %
    % Inputs:
    % PRIOR(I) = Pr(Q(1) = I)
    % TRANSMAT(I,J) = Pr(Q(T+1)=J | Q(T)=I)
    % OBSLIK(I,T) = Pr(Y(T) | Q(T)=I)
    % maximize is optional; if 1, we do max-product (as in Viterbi) instead of sum-product
    %
    % Outputs:
    % gamma(i,t) = Pr(X(t)=i | O(1:T))
    % xi(i,j,t)  = Pr(X(t)=i, X(t+1)=j | O(1:T)) t <= T-1

    if nargin < 4, maximize = 0; end

    T = size(obslik, 2);
    Q = length(prior);

    scale = ones(1,T);
    loglik = 0; 
    alpha = zeros(Q,T); 
    gamma = zeros(Q,T);
    xi = zeros(Q,Q,T-1);

    t = 1;
    alpha(:,1) = prior(:) .* obslik(:,t);
    [alpha(:,t), scale(t)] = normalise(alpha(:,t));
    transmat2 = transmat';
    for t=2:T
      if maximize
        A = repmat(alpha(:,t-1), [1 Q]);
        m = max(transmat .* A, [], 1);
        [alpha(:,t),scale(t)] = normalise(m(:) .* obslik(:,t));
      else
        [alpha(:,t),scale(t)] = normalise((transmat2 * alpha(:,t-1)) .* obslik(:,t));
      end
      if (scale(t) == 0) | isnan(scale(t)) | ~isreal(scale(t)) 
        fprintf('scale(%d)=%5.3f\n', t, scale(t))
        keyboard
      end
    end
    if any(scale==0)
      loglik = -inf;
    else
      loglik = sum(log(scale));
    end

    beta = zeros(Q,T); % beta(i,t)  = Pr(O(t+1:T) | X(t)=i)
    gamma = zeros(Q,T);
    beta(:,T) = ones(Q,1);
    gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));
    t=T;
    for t=T-1:-1:1
      b = beta(:,t+1) .* obslik(:,t+1); 
      if maximize
        B = repmat(b(:)', Q, 1);
        beta(:,t) = normalise(max(transmat .* B, [], 2));
      else
        beta(:,t) = normalise((transmat * b));
      end
      gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));
      xi(:,:,t) = normalise((transmat .* (alpha(:,t) * b')));
    end

end

end