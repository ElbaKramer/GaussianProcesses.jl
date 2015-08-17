clear all; close all;

inference = {@infDelta};
meanfunc = {@meanZero};
covfunc = {@covSum, {{@covNoise}, {@covSEiso}, {@covLinear}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covLinear}}}}};
likfunc = {@likDelta};

hyp.mean = [];
hyp.cov = [1.70386265e+00
  -5.00063816e-01
   2.49989084e+00
   3.42242275e+00
   1.94575849e+03
   2.75990985e+00
   3.79692041e+00
  -4.03397379e-01
   2.74899617e-03
  -1.95132739e+00
  -3.80566372e-01
   1.94559699e+03]';
hyp.lik = [];

load('ext/data/air.mat');
nlml = gp(hyp, inference, meanfunc, covfunc, likfunc, x, y);

numdata = 20;
for ii = 1:numdata
    
    z = linspace(min(x), max(x), size(x,1))';
    z = randsample(z, round(rand*size(z,1)));
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
    end

    [z, o] = sort(z);
    t = t(o);
    X{ii} = z;
    Y{ii} = t;

end

trial = 10;
D = zeros(trial, numdata);
for i=1:numdata
    for j=1:trial
        idx = randsample(numdata, i);
        opt = minimize(hyp, @gpall, -500, inference, meanfunc, covfunc, likfunc, X(idx), Y(idx));
        D(j,i) = norm(hyp.cov-opt.cov);
    end
end

boxplot(D);