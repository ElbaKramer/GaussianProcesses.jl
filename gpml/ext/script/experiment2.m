clear all; close all;

inference = {@infDelta};
meanfunc = {@meanZero};
covfunc = {@covSum, {{@covProd, {{@covLinear}, {@covLinear}}}, {@covProd, {{@covNoise}, {@covLinear}, {@covLinear}}}, {@covProd, {{@covSEiso}, {@covLinear}}}, {@covProd, {{@covSEiso}, {@covLinear}}}, {@covChangePointMultiD, {1, {@covSum, {{@covNoise}, {@covSEiso}}}, {@covSum, {{@covNoise}, {@covSEiso}}}}}}};
likfunc = {@likDelta};

hyp.mean = [];
hyp.cov = [  2.54824040e+00
   1.99748115e+03
  -4.44999344e+00
   2.00208988e+03
  -2.31155714e+00
  -1.43000873e+00
   2.00049071e+03
   1.95756627e+00
   2.00345431e+03
  -2.78671270e+00
  -9.34429855e-01
  -1.07584409e-01
   2.00780205e+03
  -4.64448422e+00
   3.29148115e-01
  -1.48932583e+00
   2.00460794e+03
   2.00199933e+03
   6.10122432e+00
  -2.39384193e+00
   1.78830884e+00
   3.31518158e+00
  -2.06892054e+00
   4.78400024e-01
   2.63778930e+00]';
hyp.lik = [];

load('ext/data/stock.mat');
x = X{1};
% nlml = gp(hyp, inference, meanfunc, covfunc, likfunc, x, y);

numdata = 20;
for ii = 1:numdata
    
    z = linspace(min(x), max(x), size(x,1))';
    % z = randsample(z, round(rand*size(z,1)));
    n = size(z,1);
    t = zeros(n);

    idx = [];
    for i=randperm(n)
        if isempty(idx)
            idx = i;
            m = feval(meanfunc{:}, hyp.mean, z(i));
            s2 = feval(covfunc{:}, hyp.cov, z(i));        
        else
            [m, s2] = gp(hyp, inference, meanfunc, covfunc, likfunc, z(idx), t(idx), z(i));
            idx = vertcat(idx, i);
        end
        t(i) = m + sqrt(s2)*randn;
        while t(i) < 0
            t(i) = m + sqrt(s2)*randn;
        end
    end

    [z, o] = sort(z);
    t = t(o);
    X{ii} = z;
    Y{ii} = t;

end
%%
for i=1:numdata
    figure;
    plot(X{i}, Y{i});
end

%trial = 10;
%D = zeros(trial, numdata);
%for i=1:numdata
%    for j=1:trial
%        idx = randsample(numdata, i);
%        opt = minimize(hyp, @gpall, -500, inference, meanfunc, covfunc, likfunc, X(idx), Y(idx));
%        D(j,i) = norm(hyp.cov-opt.cov);
%    end
%end

%boxplot(D);