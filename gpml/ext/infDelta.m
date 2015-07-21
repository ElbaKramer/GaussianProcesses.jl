function [post nlZ dnlZ] = infDelta(hyp, mean, cov, lik, x, y)

% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2014-12-08.
%                                      File automatically generated using noweb.
%
% See also INFMETHODS.M.

if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end
if ~strcmp(likstr,'likDelta')               % NOTE: no explicit call to likDelta
  error('Delta inference only possible with Delta likelihood');
end

[n, D] = size(x);
K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector

%sn2 = exp(hyp.lik);                                 % noise variance of likGauss

sn2 = 0;                                            % noise variance of likDelta
L = chol(K);
alpha = solve_chol(L,y-m);

%if sn2<1e-6                        % very tiny sn2 can lead to numerical trouble
%  L = chol(K+sn2*eye(n)); sl =   1;   % Cholesky factor of covariance with noise
%  pL = -solve_chol(L,eye(n));                            % L = -inv(K+inv(sW^2))
%else
%  L = chol(K/sn2+eye(n)); sl = sn2;                       % Cholesky factor of B
%  pL = L;                                           % L = chol(eye(n)+sW*sW'.*K)
%end
%alpha = solve_chol(L,y-m)/sl;

post.alpha = alpha;                            % return the posterior parameters
%post.sW = ones(n,1)/sqrt(sn2);                  % sqrt of noise precision vector
post.sW = Inf;
post.L = L;

if nargout>1                               % do we want the marginal likelihood?
  %nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sl)/2;   % -log marg lik
  nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    Q = solve_chol(L,eye(n)) - alpha*alpha';     % precompute for convenience
    %K_inv = solve_chol(L,eye(n));
    for i = 1:numel(hyp.cov)
        retval = sum(sum(Q.*feval(cov{:}, hyp.cov, x, [], i)))/2;
        if isa(retval, 'gpuArray')
            retval = gather(retval);
        end
        dnlZ.cov(i) = retval;
        %dnlZ.cov(i) = sum(sum(Q.*feval(cov{:}, hyp.cov, x, [], i)))/2;
        %dK = feval(cov{:}, hyp.cov, x, [], i);
        %dnlZ.cov(i) = 0.5*(-alpha'*dK*alpha + trace(K_inv*dK));
    end
    %dnlZ.lik = sn2*trace(Q);
    dnlZ.lik = [];
    for i = 1:numel(hyp.mean), 
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*alpha;
    end
  end
end
