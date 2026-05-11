function varargout = kreg(x,y,varargin)
% Fit a kernel regression function of y ~ x. Can use a fixed bandwidth or
% k-nearest neighbors. Can choose between several kernels. Can specify the
% limits, scale, and precision of the x-domain or can simply give the exact
% x-domain itself. Optional arugments specified using the MATLAB name-pair
% convention. Optional output arguments are returned such that one return
% argument gives the fitted function, while two return arguments are the
% [fitted function, x-domain]. The former is ideal for nesting this
% function with plot().
%
% USAGE
%   yfit = kreg(x,y);
%   [xfit, yfit] = kreg(x,y);
%   [xfit, yfit, efit] = kreg(x,y);
%   yfit = kreg(x,y,'OptionalArgName1',OptionalArgValue1,...);
%
% EXAMPLES
%   Fully specified x-domain:
%       xfit = linspace(min(x),max(x),100);
%       plot(xfit,kreg(x,y,'domain',xfit));
%
%   Using shading to specify fit error:
%       [yfit,xfit,efit] = kreg(x,y);
%       fill([xfit,fliplr(xfit)],[yfit-efit,fliplr(yfit+efit)]);
%       plot(xfit,yfit); 
%
%
% INPUT
%   x - Matrix of any number of dimensions specifying location of data on
%       the abcissa. NaN and Inf values will be ignored.
%   y - Matrix of any number of dimensions specifying location of data on
%       the ordinate. Each element in 'y' must be matched with
%       corresponding value in 'x'. NaN and Inf values will be ignored.
%
% OPTIONAL INPUT
%        bw - The fixed kernel bandwidth. Is superceded by 'knn'. If
%             neither 'bw' or 'knn' are provided, a default fixed kernel
%             bandwidth is used. The default is kernel specific:
%                   KERNEL   |   DEFAULT BW
%                  Gaussian  | Silverman's Rule of Thumb  
%                  Rectangle |  2*IQR*n^1/3 
%
%       knn - k-nearest neighbors. If all the k nearest neighbors to a 
%             particular datum have a distance of 0, then the distance to
%             the first non-zero neighbor is used.
%    domain - The exact x domain to use for random variable X. Supercedes
%             'scale', 'npoints', and 'lim'.
%     scale - The equal distance between points on an unspecified domain.
%             Mutually exclusive with 'npoints'.
%   npoints - The number of points on the unspecified domain. Mutually
%             exclusive with 'scale'.
%        xl - The [min,max] values of the unspecified domain. Default value
%             is the [min,max] of the random variable X.
%    kernel - A string specifying the type of kernel. Valid options are
%               OPTION | STRING   |   FULL NAME  |   FORM
%                  (1) |  'gauss' |  Gaussian    |    exp( -x^2 / (2*bw^2) )   default
%                  (2) | 'pgauss' |  Positive    |    exp( -x^2 / (2*bw^2) )  
%                      |          |  Gaussian    |          for x >= 0, otherwise 0
%                  (3) | 'ngauss' |  Negative    |    exp( -x^2 / (2*bw^2) )  
%                      |          |  Gaussian    |          for x <= 0, otherwise 0
%                  (4) | 'rect'   | Rectangle    |    bw/2 <= x & x <= bw/2
%
%
% OUTPUT
%   yfit - The fitted regression function.
%   xfit - The domain of the random variable X.
%   efit - The standard deviation of the regression function as a function
%          of x. Use this to shade the standard error.
%
%   
%   DHK - Jan. 22, 2022

%% manage input
x = x(:); y = y(:);
if numel(x)~=numel(y), error('numel mismatch in ''x'' and ''y''');
else
    trim = isnan(x) | isinf(x) | isnan(y) | isinf(y);
    x(trim) = [];
    y(trim) = [];
    n = numel(x);
    if n<=1, error('insufficient valid data'); end
end

ip = inputParser;
validateArg = @(arg)isnumeric(arg)&&isscalar(arg);
addOptional(ip,'bw',[],validateArg), % Kernel bandwidth
addOptional(ip,'knn',[],validateArg), % Weight data by kernel generated pdf? (true/false)
addOptional(ip,'domain',[],@(arg)isnumeric(arg)), % Exact x domain
addOptional(ip,'scale',[],validateArg), % Distance between points on x domain
addOptional(ip,'npoints',[],validateArg), % # of points on x domain
addOptional(ip,'xl',[min(x) max(x)],@(arg)isnumeric(arg)&&numel(arg)==2), % x domain boundaries [lower,upper]
addOptional(ip,'kernel','gauss',@(arg)ischar(arg)), % Kernel choice
parse(ip,varargin{:});
ip = ip.Results;

%% define kernels and bandwidths
if ip.knn > n, error('''knn'' cannot exceed the number of data'); end
if ~isempty(ip.knn) && ~isempty(ip.bw)
    warning('setting ''knn'' supercedes values set to ''bw''');
end
k = find(strcmp(ip.kernel,{'gauss','pgauss','ngauss','exp','bexp','tri','rect','skew'}));
if isempty(k), error('unsupported ''kernel'' argument. options are {''gauss'', ''rect''}'); end
% define kernel functions
switch k
    case 1
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @(x,bw) exp(-x.^2./(2.*bw.^2));
    case 2
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @(x,bw) exp(-x.^2./(2.*bw.^4)) .* (0<=x);
    case 3
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @(x,bw) exp(-x.^2./(2.*bw.^4)) .* (x<=0);
    case 4
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @(x,bw) exp(-x/bw) .* (0<=x);
    case 5
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @(x,bw) exp(-abs(x)/bw) .* (x<=0);
    case 6
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = 2*iqr(x)*n^(-1/3); end
        k = @(x,bw) (bw-abs(x))/bw .* (abs(x)<=bw);
    case 7
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = 2*iqr(x)*n^(-1/3); end
        k = @(x,bw) -bw/2 <= x & x <= bw/2;
    case 8
        if isempty(ip.bw) && isempty(ip.knn), ip.bw = .9*min([std(x) iqr(x)/1.34])*n^(-1/5); end
        k = @f;
end

%% define x domain
if ~isempty(ip.domain) % exact x domain provided by user
    if ~isempty(ip.scale) || ~isempty(ip.npoints)
        warning('setting ''domain'' supercedes values set to ''scale'' or ''npoints'''); 
    end
    xfit = ip.domain;
elseif isempty(ip.scale) && isempty(ip.npoints) % nothing specified
    xfit = linspace(ip.xl(1),ip.xl(2),100); % default to 100 point linear domain between limits of x    
elseif isempty(ip.scale) && ~isempty(ip.npoints) % npoints given
    xfit = linspace(ip.xl(1),ip.xl(2),ip.npoints); % linear domain between limits of x with n = 'npoints' points
elseif ~isempty(ip.scale) && isempty(ip.npoints) % domain scaling factor given
    xfit = ip.xl(1) : ip.scale : ip.xl(2);
else % ~isempty(ip.scale) && ~isempty(ip.npoints)
    error('not possible to set both ''scale'' and ''npoints''');
end

%% compute kernel regression
[tm,tx] = meshgrid(xfit,x);
f = tx-tm; % generate distance of predictor from kernels
clear('tm','tx');
if ~isempty(ip.knn)
    f = abs(f);
    fs = sort(f);
    ip.bw = fs(ip.knn,:); % the knn_th row
    ip.bw(ip.bw==0) = min(fs(fs>0)); % if that row contains zero, the minimum non-zero distance
    ip.bw = repmat( ip.bw*2, size(f,1),1);
end
f = k(f,ip.bw); % generate kernels
y = repmat(y,1,numel(xfit));
y = y.* f; % weight the outcome
f = sum(f);
yfit = sum(y)./f;
yfit(f==0) = 0; % protect against divide by zero nans

% [yfit,xfit,efit,f]
if 1<nargout
    varargout{1} = xfit;
    varargout{2} = yfit;
    efit = sqrt( sum( (y-repmat(yfit,size(y,1),1)).^2 ) ) ./f;
    efit(f==0) = 0;
    varargout{3} = efit;
    varargout{4} = f;
else
    varargout{1} = yfit;
end




%%
function y = iqr(x)
y = diff(prctile(x, [25, 75]));

function y = f(x,bw)
y = sknormpdf(x,0,bw,-10);
y = y./max(y(:));